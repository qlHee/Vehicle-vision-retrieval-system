#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像检索模块
实现基于特征编码的KNN图像检索
"""

import numpy as np
import pickle
import os
import time
from typing import List, Tuple, Dict, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class ImageRetriever:
    """图像检索器类"""
    
    def __init__(self, distance_metric: str = "cosine"):
        """
        初始化图像检索器
        Args:
            distance_metric: 距离度量方法 ("cosine", "euclidean", "manhattan")
        """
        self.distance_metric = distance_metric
        self.database_encodings = None
        self.database_labels = None
        self.database_paths = None
        self.knn_model = None
        
    def build_database(self, encodings: List[np.ndarray], labels: List[str], 
                      image_paths: List[str]):
        """
        构建检索数据库
        Args:
            encodings: 编码向量列表
            labels: 图像标签列表
            image_paths: 图像路径列表
        """
        print(f"正在构建检索数据库，包含{len(encodings)}张图像...")
        
        self.database_encodings = np.array(encodings)
        self.database_labels = labels
        self.database_paths = image_paths
        
        # 构建KNN模型
        if self.distance_metric == "cosine":
            # 对于余弦距离，使用cosine metric
            self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        elif self.distance_metric == "euclidean":
            self.knn_model = NearestNeighbors(metric='euclidean', algorithm='auto')
        elif self.distance_metric == "manhattan":
            self.knn_model = NearestNeighbors(metric='manhattan', algorithm='auto')
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
        
        self.knn_model.fit(self.database_encodings)
        print(f"数据库构建完成，使用{self.distance_metric}距离度量")
    
    def retrieve_knn(self, query_encoding: np.ndarray, k: int = 10) -> Dict[str, Any]:
        """
        执行KNN检索
        Args:
            query_encoding: 查询图像的编码向量
            k: 返回的近邻数量
        Returns:
            检索结果字典
        """
        if self.knn_model is None:
            raise ValueError("数据库未构建，请先调用build_database")
        
        start_time = time.time()
        
        # 执行KNN搜索
        query_encoding = query_encoding.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(query_encoding, n_neighbors=k)
        
        retrieval_time = time.time() - start_time
        
        # 整理结果
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'image_path': self.database_paths[idx],
                'label': self.database_labels[idx],
                'distance': float(dist),
                'similarity': 1.0 / (1.0 + dist) if dist > 0 else 1.0
            })
        
        return {
            'results': results,
            'retrieval_time': retrieval_time,
            'query_encoding_shape': query_encoding.shape,
            'distance_metric': self.distance_metric
        }
    
    def batch_retrieve(self, query_encodings: List[np.ndarray], k: int = 10) -> List[Dict[str, Any]]:
        """
        批量检索
        Args:
            query_encodings: 查询编码列表
            k: 返回的近邻数量
        Returns:
            检索结果列表
        """
        results = []
        for i, encoding in enumerate(query_encodings):
            print(f"检索第{i+1}/{len(query_encodings)}张图像...")
            result = self.retrieve_knn(encoding, k)
            results.append(result)
        return results
    
    def save_database(self, filepath: str):
        """保存数据库到文件"""
        data = {
            'database_encodings': self.database_encodings,
            'database_labels': self.database_labels,
            'database_paths': self.database_paths,
            'distance_metric': self.distance_metric
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据库已保存到: {filepath}")
    
    def load_database(self, filepath: str):
        """从文件加载数据库"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.database_encodings = data['database_encodings']
        self.database_labels = data['database_labels']
        self.database_paths = data['database_paths']
        self.distance_metric = data['distance_metric']
        
        # 重建KNN模型
        if self.distance_metric == "cosine":
            self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        elif self.distance_metric == "euclidean":
            self.knn_model = NearestNeighbors(metric='euclidean', algorithm='auto')
        elif self.distance_metric == "manhattan":
            self.knn_model = NearestNeighbors(metric='manhattan', algorithm='auto')
        
        self.knn_model.fit(self.database_encodings)
        print(f"数据库已从文件加载: {filepath}")

class PerformanceEvaluator:
    """性能评估器类"""
    
    def __init__(self):
        """初始化性能评估器"""
        pass
    
    def calculate_precision_recall(self, retrieved_labels: List[str], 
                                 query_label: str, k: int = 10) -> Tuple[float, float]:
        """
        计算精度和召回率
        Args:
            retrieved_labels: 检索到的标签列表
            query_label: 查询图像的真实标签
            k: 考虑的前k个结果
        Returns:
            (precision, recall)
        """
        retrieved_labels = retrieved_labels[:k]
        relevant_retrieved = sum(1 for label in retrieved_labels if label == query_label)
        
        # 精度 = 检索到的相关图像数 / 检索到的图像总数
        precision = relevant_retrieved / len(retrieved_labels) if retrieved_labels else 0.0
        
        # 召回率需要知道数据库中相关图像的总数
        # 这里假设我们有这个信息，实际使用时需要传入
        recall = relevant_retrieved / max(1, relevant_retrieved)  # 简化计算
        
        return precision, recall
    
    def calculate_average_precision(self, retrieved_labels: List[str], 
                                  query_label: str) -> float:
        """
        计算平均精度(AP)
        Args:
            retrieved_labels: 检索到的标签列表
            query_label: 查询图像的真实标签
        Returns:
            平均精度
        """
        relevant_count = 0
        precision_sum = 0.0
        
        for i, label in enumerate(retrieved_labels):
            if label == query_label:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / relevant_count
    
    def calculate_map(self, all_retrieved_labels: List[List[str]], 
                     all_query_labels: List[str]) -> float:
        """
        计算平均平均精度(MAP)
        Args:
            all_retrieved_labels: 所有查询的检索结果标签
            all_query_labels: 所有查询的真实标签
        Returns:
            MAP值
        """
        ap_sum = 0.0
        for retrieved_labels, query_label in zip(all_retrieved_labels, all_query_labels):
            ap = self.calculate_average_precision(retrieved_labels, query_label)
            ap_sum += ap
        
        return ap_sum / len(all_query_labels) if all_query_labels else 0.0
    
    def evaluate_retrieval_performance(self, retrieval_results: List[Dict[str, Any]], 
                                     query_labels: List[str], 
                                     k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        评估检索性能
        Args:
            retrieval_results: 检索结果列表
            query_labels: 查询标签列表
            k_values: 要评估的k值列表
        Returns:
            性能评估结果
        """
        print("正在评估检索性能...")
        
        # 提取检索到的标签
        all_retrieved_labels = []
        all_retrieval_times = []
        
        for result in retrieval_results:
            retrieved_labels = [r['label'] for r in result['results']]
            all_retrieved_labels.append(retrieved_labels)
            all_retrieval_times.append(result['retrieval_time'])
        
        # 计算各种指标
        performance = {}
        
        # 计算不同k值下的精度和召回率
        for k in k_values:
            precisions = []
            recalls = []
            
            for retrieved_labels, query_label in zip(all_retrieved_labels, query_labels):
                precision, recall = self.calculate_precision_recall(retrieved_labels, query_label, k)
                precisions.append(precision)
                recalls.append(recall)
            
            performance[f'precision@{k}'] = np.mean(precisions)
            performance[f'recall@{k}'] = np.mean(recalls)
        
        # 计算MAP
        performance['MAP'] = self.calculate_map(all_retrieved_labels, query_labels)
        
        # 计算平均检索时间
        performance['avg_retrieval_time'] = np.mean(all_retrieval_times)
        performance['total_retrieval_time'] = np.sum(all_retrieval_times)
        
        # 计算Top-1准确率
        top1_correct = 0
        for retrieved_labels, query_label in zip(all_retrieved_labels, query_labels):
            if retrieved_labels and retrieved_labels[0] == query_label:
                top1_correct += 1
        performance['top1_accuracy'] = top1_correct / len(query_labels) if query_labels else 0.0
        
        return performance

def test_image_retrieval():
    """测试图像检索功能"""
    print("测试图像检索功能...")
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟数据库编码
    database_encodings = [np.random.rand(256) for _ in range(100)]
    database_labels = [f"class_{i%10}" for i in range(100)]
    database_paths = [f"image_{i}.jpg" for i in range(100)]
    
    # 模拟查询编码
    query_encodings = [np.random.rand(256) for _ in range(10)]
    query_labels = [f"class_{i%10}" for i in range(10)]
    
    # 创建检索器
    retriever = ImageRetriever(distance_metric="cosine")
    retriever.build_database(database_encodings, database_labels, database_paths)
    
    # 执行检索
    print("\n执行KNN检索:")
    retrieval_results = retriever.batch_retrieve(query_encodings, k=10)
    
    # 评估性能
    evaluator = PerformanceEvaluator()
    performance = evaluator.evaluate_retrieval_performance(retrieval_results, query_labels)
    
    print("\n性能评估结果:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    test_image_retrieval()
