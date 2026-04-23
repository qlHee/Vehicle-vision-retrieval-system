#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像检索与评估模块
==================
本模块实现图像检索和性能评估功能：
1. ImageRetriever: 基于KNN的相似图像检索
2. PerformanceEvaluator: 计算检索性能指标（精度、召回率、mAP、PR曲线）
3. load_image_database: 从文件夹结构加载图像数据库

检索流程:
  查询图像编码 -> KNN搜索最近邻 -> 返回最相似的K张图像
"""

import numpy as np
import os
import time
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# 图像检索器：使用KNN算法查找相似图像
# =============================================================================
class ImageRetriever:
    """
    图像检索器类
    
    使用K近邻(KNN)算法在数据库中查找与查询图像最相似的图像。
    相似度基于图像编码向量之间的余弦距离计算。
    """
    
    def __init__(self, metric="cosine"):
        """
        初始化检索器
        
        参数:
            metric: 距离度量方式
                   - "cosine": 余弦距离，适合归一化后的向量，值域[0,2]
                   - "euclidean": 欧氏距离
        """
        self.metric = metric
        self.encodings = None  # 数据库所有图像的编码向量矩阵
        self.labels = None     # 数据库图像的标签列表（车牌号）
        self.paths = None      # 数据库图像的路径列表
        self.knn = None        # sklearn的KNN模型
    
    def build_index(self, encodings, labels, paths):
        """
        构建检索索引
        
        将所有数据库图像的编码向量加载到KNN模型中，便于后续快速检索
        
        参数:
            encodings: 编码向量列表，每个元素是一张图像的编码
            labels: 标签列表，与encodings一一对应
            paths: 路径列表，与encodings一一对应
        """
        self.encodings = np.array(encodings)
        self.labels = labels
        self.paths = paths
        # 使用暴力搜索算法（brute），对于小规模数据集足够快
        self.knn = NearestNeighbors(metric=self.metric, algorithm='brute')
        # 拟合KNN模型，即将数据库编码存入模型
        self.knn.fit(self.encodings)
        print(f"Index built: {len(encodings)} images")
    
    def search(self, query_encoding, k=10, exclude_path=None):
        """
        检索最相似的K张图像
        
        参数:
            query_encoding: 查询图像的编码向量
            k: 返回结果数量
            exclude_path: 需要排除的图像路径（用于排除查询图像本身，避免自己匹配自己）
        
        返回:
            results: 检索结果列表，每项包含 {path: 路径, label: 标签, distance: 距离}
            time_cost: 检索耗时（秒）
        """
        start = time.time()
        # 将查询向量转换为2D数组（KNN要求输入为2D）
        query = query_encoding.reshape(1, -1)
        
        # 多查询几个结果，以便排除查询图像本身后仍有足够的结果
        dists, idxs = self.knn.kneighbors(query, n_neighbors=min(k + 5, len(self.paths)))
        
        # 构建结果列表
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            path = self.paths[idx]
            # 如果是查询图像本身，跳过（通过比较文件名判断）
            if exclude_path and os.path.basename(path) == os.path.basename(exclude_path):
                continue
            results.append({'path': path, 'label': self.labels[idx], 'distance': float(dist)})
            # 收集够K个结果后停止
            if len(results) >= k:
                break
        
        return results, time.time() - start


# =============================================================================
# 性能评估器：计算检索性能指标
# =============================================================================
class PerformanceEvaluator:
    """
    性能评估器类
    
    计算图像检索的标准评估指标，用于衡量检索系统的性能：
    - Precision@K: 前K个结果中正确结果的比例
    - Recall@K: 前K个结果中正确结果占所有正确结果的比例
    - AP (Average Precision): 平均精度，综合考虑排序质量
    - mAP (mean AP): 所有查询的AP平均值
    - PR曲线: 精度-召回率曲线
    """
    
    def __init__(self, label_counts=None):
        """
        初始化评估器
        
        参数:
            label_counts: 每个标签在数据库中的图像数量 {label: count}
                         用于计算召回率时确定分母（该类别的总数）
        """
        self.label_counts = label_counts or {}
    
    # -------------------------------------------------------------------------
    # 核心评估指标
    # -------------------------------------------------------------------------
    def precision_at_k(self, retrieved, query_label, k):
        """
        计算Precision@K（精度）
        
        公式: Precision@K = (前K个结果中相关结果数) / K
        
        例如：查询车牌A，返回[A,B,A,C,A]，则P@3=2/3，P@5=3/5
        
        参数:
            retrieved: 检索返回的标签列表，按相似度排序
            query_label: 查询图像的真实标签
            k: 考虑前K个结果
        
        返回:
            精度值，范围[0,1]
        """
        top_k = retrieved[:k]
        # 统计前K个结果中与查询标签相同的数量
        relevant = sum(1 for lbl in top_k if lbl == query_label)
        return relevant / k if k > 0 else 0.0
    
    def recall_at_k(self, retrieved, query_label, k):
        """
        计算Recall@K（召回率）
        
        公式: Recall@K = (前K个结果中相关结果数) / (数据库中该类别总数)
        
        衡量的是"找全了多少"，即相关图像有多少被检索出来了
        
        参数:
            retrieved: 检索返回的标签列表
            query_label: 查询图像的真实标签
            k: 考虑前K个结果
        
        返回:
            召回率值，范围[0,1]
        """
        top_k = retrieved[:k]
        relevant = sum(1 for lbl in top_k if lbl == query_label)
        # 数据库中该类别的总数，减1是因为要排除查询图像本身
        total = self.label_counts.get(query_label, 1) - 1
        return relevant / total if total > 0 else 0.0
    
    def average_precision(self, retrieved, query_label, query_in_database=True):
        """
        计算AP（Average Precision，平均精度）
        
        AP综合考虑了检索结果的排序质量：
        - 不仅看找到了多少正确结果，还看正确结果排在什么位置
        - 正确结果排得越靠前，AP越高
        
        公式: AP = (1/R) * Σ(Precision@i * rel(i))
        其中R是相关文档总数，rel(i)表示第i个结果是否相关
        
        例如：查询A，返回[A,B,A]
        - 位置1是A，此时P@1=1/1=1
        - 位置3是A，此时P@3=2/3
        - AP = (1+2/3) / 2 = 0.833
        
        参数:
            retrieved: 检索返回的标签列表
            query_label: 查询图像的真实标签
            query_in_database: 查询图像是否在数据库中（若是则需排除自身）
        
        返回:
            AP值，范围[0,1]
        """
        prec_sum = 0.0   # 精度累加和
        rel_count = 0    # 累计遇到的相关结果数
        
        for i, lbl in enumerate(retrieved):
            if lbl == query_label:
                rel_count += 1
                # 在这个位置的精度
                prec_sum += rel_count / (i + 1)
        
        # 分母使用 min(相关文档总数, K)，因为在前K个位置最多只能找到这么多相关文档
        total_relevant = self.label_counts.get(query_label, 0)
        if query_in_database:
            total_relevant -= 1
        # AP@K 的分母应为 min(R, K)，其中 K = len(retrieved)
        denominator = min(total_relevant, len(retrieved)) if total_relevant > 0 else 0
        return prec_sum / denominator if denominator > 0 else 0.0
    
    def mean_average_precision(self, all_retrieved, all_queries, k, query_in_database=True):
        """
        计算mAP@K（mean Average Precision）
        
        mAP是所有查询的AP的平均值，是衡量检索系统整体性能的重要指标
        
        参数:
            all_retrieved: 所有查询的检索结果标签列表
            all_queries: 所有查询的真实标签列表
            k: 只考虑每个查询的前K个结果
            query_in_database: 查询图像是否在数据库中（若是则需排除自身）
        
        返回:
            mAP值，范围[0,1]
        """
        # 对每个查询计算AP（只考虑前K个结果），然后取平均
        ap_sum = sum(self.average_precision(ret[:k], q, query_in_database) for ret, q in zip(all_retrieved, all_queries))
        return ap_sum / len(all_queries) if all_queries else 0.0
    
    # -------------------------------------------------------------------------
    # PR曲线计算
    # -------------------------------------------------------------------------
    def compute_pr_curve(self, all_retrieved, all_queries):
        """
        计算PR曲线的数据点
        
        PR曲线展示精度(Precision)与召回率(Recall)的权衡关系：
        - 横轴是Recall，纵轴是Precision
        - 理想情况下曲线越靠近右上角越好
        - 曲线下面积(AUC)可以用来比较不同算法的性能
        
        参数:
            all_retrieved: 所有查询的检索结果标签列表
            all_queries: 所有查询的真实标签列表
        
        返回:
            (precisions, recalls): 两个列表，分别是不同K值下的平均精度和平均召回率
        """
        # 找出最长的结果列表长度，决定K的范围
        max_k = max(len(r) for r in all_retrieved) if all_retrieved else 0
        precisions, recalls = [], []
        
        # 对每个K值，计算所有查询的平均精度和平均召回率
        for k in range(1, max_k + 1):
            p = np.mean([self.precision_at_k(r, q, k) for r, q in zip(all_retrieved, all_queries)])
            r = np.mean([self.recall_at_k(r, q, k) for r, q in zip(all_retrieved, all_queries)])
            precisions.append(p)
            recalls.append(r)
        
        return precisions, recalls
    
    # -------------------------------------------------------------------------
    # 综合评估
    # -------------------------------------------------------------------------
    def evaluate(self, retrieval_results, k_values=list(range(1, 11)), query_in_database=False):
        """
        综合评估检索性能
        
        对一组检索结果计算所有指标，返回完整的评估报告
        
        参数:
            retrieval_results: 检索结果列表，每项是 (搜索结果列表, 查询标签) 的元组
            k_values: 要计算的K值列表，默认1到10
            query_in_database: 查询图像是否在数据库中（测试集评估时为False）
        
        返回:
            metrics: 评估指标字典，包含：
                    - Precision@K, Recall@K, mAP@K（对每个K值）
                    - pr_curve: PR曲线数据
        """
        # 提取所有检索结果的标签列表
        all_retrieved = [[r['label'] for r in res] for res, _ in retrieval_results]
        # 提取所有查询的真实标签
        all_queries = [lbl for _, lbl in retrieval_results]
        
        metrics = {}
        # 对每个K值计算指标
        for k in k_values:
            metrics[f'Precision@{k}'] = np.mean([self.precision_at_k(r, q, k) for r, q in zip(all_retrieved, all_queries)])
            metrics[f'Recall@{k}'] = np.mean([self.recall_at_k(r, q, k) for r, q in zip(all_retrieved, all_queries)])
            metrics[f'mAP@{k}'] = self.mean_average_precision(all_retrieved, all_queries, k, query_in_database)
        
        # 计算PR曲线数据
        prec, rec = self.compute_pr_curve(all_retrieved, all_queries)
        metrics['pr_curve'] = {'precision': prec, 'recall': rec}
        
        return metrics


# =============================================================================
# 工具函数：从文件夹结构加载图像数据库
# =============================================================================
def load_image_database(image_folder):
    """
    加载图像数据库
    
    假设图像按照以下结构组织：
    image_folder/
    ├── 车牌号1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── 车牌号2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    
    每个子文件夹的名称就是该文件夹内所有图像的标签（车牌号）
    
    参数:
        image_folder: 图像根目录路径
    
    返回:
        database: 字典 {标签: [图像路径列表]}
    """
    database = {}
    # 遍历根目录下的所有子文件夹
    for label in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label)
        # 跳过非目录和隐藏文件夹（以.开头）
        if os.path.isdir(label_path) and not label.startswith('.'):
            # 获取该文件夹下所有图像文件
            images = [os.path.join(label_path, f) for f in os.listdir(label_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                database[label] = images
    return database
