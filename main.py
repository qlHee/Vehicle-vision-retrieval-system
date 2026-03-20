#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab2主程序：特征编码与图像检索系统
基于Lab1的特征提取结果，实现特征编码、字典构建和图像检索
"""

import cv2
import numpy as np
import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple
import argparse

# 添加Lab1路径以导入特征提取模块
sys.path.append('/Users/tuxol/Documents/DataVault/CST/CV/Lab1')
from feature_extraction import FeatureExtractor

# 导入Lab2模块
from feature_encoding import FeatureEncoder
from image_retrieval import ImageRetriever, PerformanceEvaluator

class Lab2System:
    """Lab2特征编码与图像检索系统"""
    
    def __init__(self, image_folder: str, dictionary_size: int = 256):
        """
        初始化系统
        Args:
            image_folder: 图像文件夹路径
            dictionary_size: 字典大小
        """
        self.image_folder = image_folder
        self.dictionary_size = dictionary_size
        
        # 初始化各个组件
        self.feature_extractor = FeatureExtractor()
        self.feature_encoder = FeatureEncoder(dictionary_size)
        self.image_retriever = ImageRetriever()
        self.performance_evaluator = PerformanceEvaluator()
        
        # 数据存储
        self.image_database = {}
        self.train_encodings = []
        self.train_labels = []
        self.train_paths = []
        
    def load_images_and_extract_features(self, algorithm: str = "SIFT"):
        """
        加载图像并提取特征
        Args:
            algorithm: 特征提取算法 ("SIFT", "KAZE", "ORB")
        """
        print(f"正在加载图像并使用{algorithm}提取特征...")
        
        image_count = 0
        for category_folder in os.listdir(self.image_folder):
            category_path = os.path.join(self.image_folder, category_folder)
            if os.path.isdir(category_path) and not category_folder.startswith('.'):
                print(f"处理类别: {category_folder}")
                
                for image_file in os.listdir(category_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(category_path, image_file)
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                # 提取特征
                                features = self.feature_extractor.extract_and_time(image, algorithm)
                                
                                if features['descriptors'] is not None:
                                    self.image_database[image_path] = {
                                        'image': image,
                                        'category': category_folder,
                                        'features': features,
                                        'descriptors': features['descriptors']
                                    }
                                    image_count += 1
                                    
                        except Exception as e:
                            print(f"处理图像失败 {image_path}: {e}")
        
        print(f"成功加载并提取特征的图像数量: {image_count}")
        return image_count
    
    def build_dictionary_and_encode(self, encoding_method: str = "BoF", 
                                   dictionary_method: str = "kmeans"):
        """
        构建字典并编码所有图像
        Args:
            encoding_method: 编码方法 ("BoF", "VLAD", "FV")
            dictionary_method: 字典构建方法 ("kmeans", "gmm")
        """
        print(f"正在构建字典并使用{encoding_method}方法编码图像...")
        
        # 收集所有描述子用于构建字典
        all_descriptors = []
        for image_data in self.image_database.values():
            if image_data['descriptors'] is not None:
                all_descriptors.append(image_data['descriptors'])
        
        if not all_descriptors:
            raise ValueError("没有有效的描述子用于构建字典")
        
        # 构建字典
        if dictionary_method == "kmeans" or encoding_method in ["BoF", "VLAD"]:
            self.feature_encoder.build_dictionary_kmeans(all_descriptors)
        elif dictionary_method == "gmm" or encoding_method == "FV":
            self.feature_encoder.build_dictionary_gmm(all_descriptors)
        
        # 编码所有图像
        print("正在编码所有图像...")
        encoded_count = 0
        
        for image_path, image_data in self.image_database.items():
            try:
                # 编码图像
                encoding = self.feature_encoder.encode_image(
                    image_data['descriptors'], encoding_method
                )
                
                # 存储编码结果
                image_data['encoding'] = encoding
                self.train_encodings.append(encoding)
                self.train_labels.append(image_data['category'])
                self.train_paths.append(image_path)
                
                encoded_count += 1
                
            except Exception as e:
                print(f"编码图像失败 {image_path}: {e}")
        
        print(f"成功编码的图像数量: {encoded_count}")
        
        # 构建检索数据库
        self.image_retriever.build_database(
            self.train_encodings, self.train_labels, self.train_paths
        )
    
    def split_train_test(self, test_ratio: float = 0.2) -> Tuple[List, List]:
        """
        分割训练集和测试集
        Args:
            test_ratio: 测试集比例
        Returns:
            (train_indices, test_indices)
        """
        # 按类别分割
        category_indices = {}
        for i, (image_path, image_data) in enumerate(self.image_database.items()):
            category = image_data['category']
            if category not in category_indices:
                category_indices[category] = []
            category_indices[category].append(i)
        
        train_indices = []
        test_indices = []
        
        for category, indices in category_indices.items():
            np.random.shuffle(indices)
            split_point = int(len(indices) * (1 - test_ratio))
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])
        
        print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")
        return train_indices, test_indices
    
    def evaluate_system(self, test_indices: List[int], k: int = 10) -> Dict[str, Any]:
        """
        评估系统性能
        Args:
            test_indices: 测试集索引
            k: KNN的k值
        Returns:
            评估结果
        """
        print(f"正在评估系统性能，测试集大小: {len(test_indices)}")
        
        # 准备测试数据
        test_encodings = []
        test_labels = []
        test_paths = []
        
        image_paths = list(self.image_database.keys())
        for idx in test_indices:
            image_path = image_paths[idx]
            image_data = self.image_database[image_path]
            test_encodings.append(image_data['encoding'])
            test_labels.append(image_data['category'])
            test_paths.append(image_path)
        
        # 执行检索
        retrieval_results = self.image_retriever.batch_retrieve(test_encodings, k)
        
        # 评估性能
        performance = self.performance_evaluator.evaluate_retrieval_performance(
            retrieval_results, test_labels, k_values=[1, 5, 10]
        )
        
        return performance, retrieval_results
    
    def save_results(self, performance: Dict[str, Any], 
                    encoding_method: str, algorithm: str):
        """
        保存实验结果
        Args:
            performance: 性能评估结果
            encoding_method: 编码方法
            algorithm: 特征提取算法
        """
        # 保存性能结果
        results = {
            'algorithm': algorithm,
            'encoding_method': encoding_method,
            'dictionary_size': self.dictionary_size,
            'performance': performance,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_file = f"results/performance_{algorithm}_{encoding_method}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"性能结果已保存到: {results_file}")
        
        # 保存字典和数据库
        dictionary_file = f"models/dictionary_{algorithm}_{encoding_method}.pkl"
        self.feature_encoder.save_dictionary(dictionary_file)
        
        database_file = f"models/database_{algorithm}_{encoding_method}.pkl"
        self.image_retriever.save_database(database_file)
    
    def run_experiment(self, algorithm: str = "SIFT", 
                      encoding_method: str = "BoF",
                      test_ratio: float = 0.2,
                      k: int = 10):
        """
        运行完整实验
        Args:
            algorithm: 特征提取算法
            encoding_method: 编码方法
            test_ratio: 测试集比例
            k: KNN的k值
        """
        print(f"开始运行实验: {algorithm} + {encoding_method}")
        print("=" * 60)
        
        # 步骤1: 加载图像并提取特征
        self.load_images_and_extract_features(algorithm)
        
        # 步骤2: 构建字典并编码
        self.build_dictionary_and_encode(encoding_method)
        
        # 步骤3: 分割训练测试集
        train_indices, test_indices = self.split_train_test(test_ratio)
        
        # 步骤4: 评估性能
        performance, retrieval_results = self.evaluate_system(test_indices, k)
        
        # 步骤5: 保存结果
        self.save_results(performance, encoding_method, algorithm)
        
        # 打印结果
        print("\n实验结果:")
        print("-" * 40)
        for metric, value in performance.items():
            print(f"{metric}: {value:.4f}")
        
        return performance, retrieval_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Lab2特征编码与图像检索系统')
    parser.add_argument('--image_folder', type=str, 
                       default='/Users/tuxol/Documents/DataVault/CST/CV/Lab1/image',
                       help='图像文件夹路径')
    parser.add_argument('--algorithm', type=str, default='SIFT',
                       choices=['SIFT', 'KAZE', 'ORB'],
                       help='特征提取算法')
    parser.add_argument('--encoding', type=str, default='BoF',
                       choices=['BoF', 'VLAD', 'FV'],
                       help='特征编码方法')
    parser.add_argument('--dictionary_size', type=int, default=256,
                       help='字典大小')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--k', type=int, default=10,
                       help='KNN的k值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # 创建系统实例
    system = Lab2System(args.image_folder, args.dictionary_size)
    
    # 运行实验
    try:
        performance, retrieval_results = system.run_experiment(
            algorithm=args.algorithm,
            encoding_method=args.encoding,
            test_ratio=args.test_ratio,
            k=args.k
        )
        
        print(f"\n实验完成！结果已保存到results/目录")
        
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
