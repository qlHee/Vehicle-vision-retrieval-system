#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆视觉检索系统 - 主程序入口
==============================
本程序是车辆视觉检索系统的主入口，支持两种运行模式：
1. GUI模式: 启动图形界面，交互式进行图像检索
2. 评估模式: 命令行批量评估，对比BoF/VLAD和TF-IDF的效果

使用方法:
    python main.py --mode gui     # 启动图形界面
    python main.py --mode eval    # 运行对比评估实验
    python main.py --mode eval --output ./my_results  # 指定输出目录
"""

import os
import cv2
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，支持无显示器环境
import matplotlib.pyplot as plt

from feature_extractor import FeatureExtractor, FeatureEncoder
from image_retrieval import ImageRetriever, PerformanceEvaluator, load_image_database


# =============================================================================
# 命令行评估模式的核心系统类
# =============================================================================
class ImageSearchSystem:
    """
    图像检索系统核心类（用于命令行评估模式）
    
    封装了完整的检索流程：
    1. 加载图像并提取特征
    2. 构建视觉词典
    3. 编码所有图像
    4. 在测试集上评估性能
    """
    
    def __init__(self, image_folder, test_folder, codebook_size=256):
        """
        初始化检索系统
        
        参数:
            image_folder: 数据库图像文件夹路径
            test_folder: 测试集图像文件夹路径
            codebook_size: 视觉词典大小，默认256
        """
        self.image_folder = image_folder
        self.test_folder = test_folder
        
        # 初始化核心组件
        self.extractor = FeatureExtractor()           # 特征提取器
        self.encoder = FeatureEncoder(codebook_size)  # 特征编码器
        self.retriever = ImageRetriever(metric="cosine")  # 图像检索器
        
        # 数据存储
        self.database = {}          # 数据库信息 {路径: {label, descriptors}}
        self.all_descriptors = []   # 所有图像的描述子
        self.all_labels = []        # 所有图像的标签
        self.all_paths = []         # 所有图像的路径
        self.evaluator = None       # 性能评估器
    
    def load_and_extract(self, algorithm="SIFT"):
        """
        步骤1：加载数据库图像并提取特征
        
        参数:
            algorithm: 特征提取算法，"SIFT"或"ORB"
        """
        print(f"[1/4] Extracting {algorithm} features...")
        db = load_image_database(self.image_folder)
        
        # 清空之前的数据
        self.database.clear()
        self.all_descriptors, self.all_labels, self.all_paths = [], [], []
        
        # 遍历所有图像，提取特征
        total = sum(len(imgs) for imgs in db.values())
        for count, (label, paths) in enumerate(db.items()):
            for path in paths:
                img = cv2.imread(path)
                if img is None:
                    continue
                # 提取特征
                _, desc, _ = self.extractor.extract(img, algorithm)
                self.database[path] = {'label': label, 'descriptors': desc}
                if desc is not None:
                    self.all_descriptors.append(desc)
                    self.all_labels.append(label)
                    self.all_paths.append(path)
        
        print(f"  Loaded {len(self.all_paths)} images")
        # 统计每个类别的图像数量，用于计算召回率
        label_counts = {l: self.all_labels.count(l) for l in set(self.all_labels)}
        self.evaluator = PerformanceEvaluator(label_counts)
    
    def build_codebook_and_encode(self, method="BoF", use_idf=False):
        """
        步骤2-4：构建视觉词典并编码所有图像
        
        参数:
            method: 编码方法，"BoF"或"VLAD"
            use_idf: 是否使用TF-IDF加权
        """
        # 步骤2：使用K-means聚类构建视觉词典
        print("[2/4] Building codebook...")
        self.encoder.build_codebook(self.all_descriptors)
        
        # 步骤3：计算IDF权重（可选）
        if use_idf:
            print("[3/4] Computing IDF...")
            self.encoder.compute_idf(self.all_descriptors)
        else:
            print("[3/4] Skipping IDF...")
        
        # 步骤4：编码所有数据库图像并构建检索索引
        print(f"[4/4] Encoding with {method}...")
        encodings = [self.encoder.encode(self.database[p]['descriptors'], method, use_idf) 
                     for p in self.all_paths]
        self.retriever.build_index(encodings, self.all_labels, self.all_paths)
    
    def evaluate(self, method="BoF", use_idf=False):
        """
        在测试集上评估检索性能
        
        参数:
            method: 编码方法
            use_idf: 是否使用IDF加权
        
        返回:
            metrics: 评估指标字典，包含Precision、Recall、mAP、检索时间等
        """
        print("Evaluating...")
        test_db = load_image_database(self.test_folder)
        
        results_list, times = [], []
        # 对测试集中的每张图像执行检索
        for label, paths in test_db.items():
            for path in paths:
                img = cv2.imread(path)
                if img is None:
                    continue
                # 提取特征并编码
                _, desc, _ = self.extractor.extract(img, "SIFT")
                enc = self.encoder.encode(desc, method, use_idf)
                # 执行检索
                results, t = self.retriever.search(enc, k=10, exclude_path=path)
                results_list.append((results, label))
                times.append(t)
        
        # 计算评估指标
        metrics = self.evaluator.evaluate(results_list)
        metrics['avg_time_ms'] = np.mean(times) * 1000  # 平均检索时间
        metrics['total_queries'] = len(results_list)    # 查询总数
        return metrics


# =============================================================================
# 对比实验
# =============================================================================
def run_comparison_experiment(image_folder, test_folder, output_dir):
    """
    运行对比实验
    
    对比4种编码组合的检索性能：
    1. BoF（不使用IDF）
    2. BoF + TF-IDF
    3. VLAD（不使用IDF）
    4. VLAD + IDF
    
    参数:
        image_folder: 数据库图像文件夹
        test_folder: 测试集文件夹
        output_dir: 结果输出目录
    
    返回:
        results: 各方法的评估结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 4种实验配置：(编码方法, 是否使用IDF, 显示名称)
    configs = [("BoF", False, "BoF"), ("BoF", True, "BoF+IDF"),
               ("VLAD", False, "VLAD"), ("VLAD", True, "VLAD+IDF")]
    
    # 初始化系统并提取特征（只需做一次）
    system = ImageSearchSystem(image_folder, test_folder)
    system.load_and_extract("SIFT")
    
    print("\n" + "=" * 50 + "\nComparison Experiment\n" + "=" * 50)
    
    # 对每种配置进行评估
    results = {}
    for method, use_idf, name in configs:
        print(f"\n>>> {name}")
        system.build_codebook_and_encode(method, use_idf)
        metrics = system.evaluate(method, use_idf)
        results[name] = metrics
        # 打印关键指标
        print(f"  mAP@10: {metrics['mAP@10']*100:.1f}% | P@1: {metrics['Precision@1']*100:.1f}% | Time: {metrics['avg_time_ms']:.1f}ms")
    
    # 生成可视化图表和报告
    _plot_comparison(results, output_dir)   # 精度/召回率/时间对比柱状图
    _plot_pr_curves(results, output_dir)    # PR曲线对比
    _plot_map_curves(results, output_dir)   # mAP@K曲线对比
    _save_report(results, output_dir)       # 文本报告
    
    print(f"\nDone! Results saved to: {output_dir}")
    return results


# =============================================================================
# 可视化绘图函数
# =============================================================================
def _plot_comparison(results, output_dir):
    """
    绘制性能对比柱状图
    
    生成包含三个子图的图表：
    1. Precision@1/5/10对比
    2. Recall@1/5/10对比
    3. 平均检索时间对比
    """
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x, width = np.arange(len(methods)), 0.25
    
    # 子图1：Precision对比
    for i, k in enumerate([1, 5, 10]):
        axes[0].bar(x + i*width, [results[m][f'Precision@{k}']*100 for m in methods], width, label=f'P@{k}')
    axes[0].set_ylabel('Precision (%)')
    axes[0].set_title('Precision')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(methods, rotation=15)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 子图2：Recall对比
    for i, k in enumerate([1, 5, 10]):
        axes[1].bar(x + i*width, [results[m][f'Recall@{k}']*100 for m in methods], width, label=f'R@{k}')
    axes[1].set_ylabel('Recall (%)')
    axes[1].set_title('Recall')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(methods, rotation=15)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # 子图3：检索时间对比
    axes[2].bar(methods, [results[m]['avg_time_ms'] for m in methods], color='steelblue')
    axes[2].set_ylabel('Time (ms)')
    axes[2].set_title('Avg Retrieval Time')
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()


def _plot_pr_curves(results, output_dir):
    """
    绘制PR曲线对比图
    
    在同一张图上绘制所有方法的PR曲线，便于直观比较
    曲线越靠近右上角说明性能越好
    """
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for (name, m), c in zip(results.items(), colors):
        plt.plot(m['pr_curve']['recall'], m['pr_curve']['precision'], c, lw=2, label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150)
    plt.close()


def _plot_map_curves(results, output_dir):
    """
    绘制mAP@K曲线对比图
    
    展示各方法在不同K值下的mAP变化趋势
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for (name, m), c in zip(results.items(), colors):
        # 绘制K=1到K=10的mAP曲线
        plt.plot(range(1, 11), [m[f'mAP@{k}']*100 for k in range(1, 11)], c, lw=2, marker='o', label=name)
    plt.xlabel('K')
    plt.ylabel('mAP (%)')
    plt.title('mAP@K')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_comparison.png'), dpi=150)
    plt.close()


def _save_report(results, output_dir):
    """
    保存评估报告到文本文件
    
    报告内容包括：
    - 各方法的详细指标（Precision、Recall、mAP）
    - 最佳方法总结
    """
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write("=" * 50 + "\nEvaluation Report\n" + "=" * 50 + "\n\n")
        
        # 输出每种方法的详细指标
        for name, m in results.items():
            f.write(f"--- {name} ---\n")
            f.write(f"Queries: {m['total_queries']} | Time: {m['avg_time_ms']:.1f}ms\n")
            for k in [1, 5, 10]:
                f.write(f"  @{k}: P={m[f'Precision@{k}']*100:.1f}% R={m[f'Recall@{k}']*100:.1f}% mAP={m[f'mAP@{k}']*100:.1f}%\n")
            f.write("\n")
        
        # 找出最佳方法并总结
        best_map = max(results.items(), key=lambda x: x[1]['mAP@10'])
        best_p1 = max(results.items(), key=lambda x: x[1]['Precision@1'])
        f.write(f"Best mAP@10: {best_map[0]} ({best_map[1]['mAP@10']*100:.1f}%)\n")
        f.write(f"Best P@1: {best_p1[0]} ({best_p1[1]['Precision@1']*100:.1f}%)\n")


# =============================================================================
# 主程序入口
# =============================================================================
def main():
    """
    主函数：解析命令行参数，启动相应模式
    
    支持的参数：
        --mode: 运行模式，"gui"（图形界面）或"eval"（命令行评估）
        --image_folder: 数据库图像文件夹路径（可选）
        --test_folder: 测试集文件夹路径（可选）
        --output: 评估结果输出目录（可选，默认./results）
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Vehicle Image Retrieval System')
    parser.add_argument('--mode', default='gui', choices=['gui', 'eval'], help='gui or eval')
    parser.add_argument('--image_folder', help='Database folder')
    parser.add_argument('--test_folder', help='Test folder')
    parser.add_argument('--output', default='./results', help='Output directory')
    args = parser.parse_args()
    
    # 确定数据文件夹路径（如果未指定，使用默认的上级目录结构）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    image_folder = args.image_folder or os.path.join(base_dir, "image")
    test_folder = args.test_folder or os.path.join(base_dir, "test")
    
    # 根据模式启动相应功能
    if args.mode == 'gui':
        print("Launching GUI...")
        from gui_app import main as gui_main
        gui_main()
    else:
        print("Running evaluation...")
        run_comparison_experiment(image_folder, test_folder, args.output)


if __name__ == "__main__":
    main()
