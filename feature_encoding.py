#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征编码模块
实现BoF、VLAD、FV等特征编码算法
"""

import cv2
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Dict, Any
import time

class FeatureEncoder:
    """特征编码器类"""
    
    def __init__(self, dictionary_size: int = 256):
        """
        初始化特征编码器
        Args:
            dictionary_size: 字典大小（码本大小）
        """
        self.dictionary_size = dictionary_size
        self.dictionary = None
        self.gmm_model = None
        self.encoding_method = "BoF"
        
    def build_dictionary_kmeans(self, descriptors_list: List[np.ndarray]) -> np.ndarray:
        """
        使用K-means算法构建视觉词典
        Args:
            descriptors_list: 描述子列表
        Returns:
            字典（聚类中心）
        """
        print(f"正在使用K-means构建大小为{self.dictionary_size}的视觉词典...")
        
        # 合并所有描述子
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
        
        if not all_descriptors:
            raise ValueError("没有有效的描述子用于构建字典")
        
        all_descriptors = np.vstack(all_descriptors)
        print(f"总描述子数量: {len(all_descriptors)}")
        
        # 使用K-means聚类
        start_time = time.time()
        kmeans = KMeans(n_clusters=self.dictionary_size, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)
        
        self.dictionary = kmeans.cluster_centers_
        build_time = time.time() - start_time
        print(f"字典构建完成，耗时: {build_time:.2f}秒")
        
        return self.dictionary
    
    def build_dictionary_gmm(self, descriptors_list: List[np.ndarray]) -> GaussianMixture:
        """
        使用高斯混合模型构建字典（用于Fisher Vector）
        Args:
            descriptors_list: 描述子列表
        Returns:
            GMM模型
        """
        print(f"正在使用GMM构建大小为{self.dictionary_size}的字典...")
        
        # 合并所有描述子
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
        
        if not all_descriptors:
            raise ValueError("没有有效的描述子用于构建字典")
        
        all_descriptors = np.vstack(all_descriptors)
        print(f"总描述子数量: {len(all_descriptors)}")
        
        # 使用GMM
        start_time = time.time()
        self.gmm_model = GaussianMixture(n_components=self.dictionary_size, random_state=42)
        self.gmm_model.fit(all_descriptors)
        
        build_time = time.time() - start_time
        print(f"GMM字典构建完成，耗时: {build_time:.2f}秒")
        
        return self.gmm_model
    
    def encode_bof(self, descriptors: np.ndarray) -> np.ndarray:
        """
        使用Bag of Features编码
        Args:
            descriptors: 图像描述子
        Returns:
            BoF编码向量
        """
        if self.dictionary is None:
            raise ValueError("字典未构建，请先调用build_dictionary_kmeans")
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.dictionary_size)
        
        # 计算每个描述子到字典中心的距离
        distances = np.sqrt(((descriptors - self.dictionary[:, np.newaxis])**2).sum(axis=2))
        
        # 找到最近的聚类中心
        closest_clusters = np.argmin(distances, axis=0)
        
        # 构建直方图
        histogram = np.bincount(closest_clusters, minlength=self.dictionary_size)
        
        # 归一化
        if np.sum(histogram) > 0:
            histogram = histogram.astype(np.float32) / np.sum(histogram)
        
        return histogram
    
    def encode_vlad(self, descriptors: np.ndarray) -> np.ndarray:
        """
        使用VLAD (Vector of Locally Aggregated Descriptors)编码
        Args:
            descriptors: 图像描述子
        Returns:
            VLAD编码向量
        """
        if self.dictionary is None:
            raise ValueError("字典未构建，请先调用build_dictionary_kmeans")
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.dictionary_size * descriptors.shape[1] if descriptors is not None else self.dictionary_size * 128)
        
        # 计算每个描述子到字典中心的距离
        distances = np.sqrt(((descriptors - self.dictionary[:, np.newaxis])**2).sum(axis=2))
        
        # 找到最近的聚类中心
        closest_clusters = np.argmin(distances, axis=0)
        
        # 初始化VLAD向量
        vlad_vector = np.zeros((self.dictionary_size, descriptors.shape[1]))
        
        # 对每个聚类中心计算残差累积
        for i in range(self.dictionary_size):
            # 找到分配给第i个聚类中心的描述子
            mask = closest_clusters == i
            if np.any(mask):
                # 计算残差并累积
                residuals = descriptors[mask] - self.dictionary[i]
                vlad_vector[i] = np.sum(residuals, axis=0)
        
        # 展平并L2归一化
        vlad_vector = vlad_vector.flatten()
        norm = np.linalg.norm(vlad_vector)
        if norm > 0:
            vlad_vector = vlad_vector / norm
        
        return vlad_vector
    
    def encode_fisher_vector(self, descriptors: np.ndarray) -> np.ndarray:
        """
        使用Fisher Vector编码
        Args:
            descriptors: 图像描述子
        Returns:
            Fisher Vector编码
        """
        if self.gmm_model is None:
            raise ValueError("GMM模型未构建，请先调用build_dictionary_gmm")
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(2 * self.dictionary_size * descriptors.shape[1] if descriptors is not None else 2 * self.dictionary_size * 128)
        
        # 计算后验概率
        posteriors = self.gmm_model.predict_proba(descriptors)
        
        # 获取GMM参数
        means = self.gmm_model.means_
        covariances = self.gmm_model.covariances_
        weights = self.gmm_model.weights_
        
        # 初始化Fisher向量
        d = descriptors.shape[1]
        fv = np.zeros(2 * self.dictionary_size * d)
        
        # 计算Fisher向量
        for k in range(self.dictionary_size):
            # 计算统计量
            gamma_k = posteriors[:, k]
            sum_gamma_k = np.sum(gamma_k)
            
            if sum_gamma_k > 0:
                # 一阶统计量（均值偏差）
                diff_mean = descriptors - means[k]
                weighted_diff = gamma_k[:, np.newaxis] * diff_mean
                first_order = np.sum(weighted_diff, axis=0) / (np.sqrt(weights[k]) * sum_gamma_k)
                
                # 二阶统计量（方差偏差）
                if len(covariances.shape) == 3:  # full covariance
                    sigma_k = np.diag(covariances[k])
                else:  # diagonal covariance
                    sigma_k = covariances[k]
                
                second_order_term = (diff_mean**2 / sigma_k - 1) / np.sqrt(2 * weights[k])
                weighted_second = gamma_k[:, np.newaxis] * second_order_term
                second_order = np.sum(weighted_second, axis=0) / sum_gamma_k
                
                # 填充Fisher向量
                fv[k*d:(k+1)*d] = first_order
                fv[(self.dictionary_size + k)*d:(self.dictionary_size + k + 1)*d] = second_order
        
        # L2归一化
        norm = np.linalg.norm(fv)
        if norm > 0:
            fv = fv / norm
        
        return fv
    
    def encode_image(self, descriptors: np.ndarray, method: str = "BoF") -> np.ndarray:
        """
        对图像进行编码
        Args:
            descriptors: 图像描述子
            method: 编码方法 ("BoF", "VLAD", "FV")
        Returns:
            编码向量
        """
        self.encoding_method = method
        
        if method == "BoF":
            return self.encode_bof(descriptors)
        elif method == "VLAD":
            return self.encode_vlad(descriptors)
        elif method == "FV":
            return self.encode_fisher_vector(descriptors)
        else:
            raise ValueError(f"不支持的编码方法: {method}")
    
    def save_dictionary(self, filepath: str):
        """保存字典到文件"""
        data = {
            'dictionary': self.dictionary,
            'gmm_model': self.gmm_model,
            'dictionary_size': self.dictionary_size,
            'encoding_method': self.encoding_method
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"字典已保存到: {filepath}")
    
    def load_dictionary(self, filepath: str):
        """从文件加载字典"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.dictionary = data['dictionary']
        self.gmm_model = data['gmm_model']
        self.dictionary_size = data['dictionary_size']
        self.encoding_method = data.get('encoding_method', 'BoF')
        print(f"字典已从文件加载: {filepath}")

def test_feature_encoding():
    """测试特征编码功能"""
    print("测试特征编码功能...")
    
    # 创建模拟数据
    np.random.seed(42)
    descriptors_list = [np.random.rand(100, 128) for _ in range(10)]
    test_descriptors = np.random.rand(50, 128)
    
    # 创建编码器
    encoder = FeatureEncoder(dictionary_size=64)
    
    # 测试BoF编码
    print("\n测试BoF编码:")
    encoder.build_dictionary_kmeans(descriptors_list)
    bof_encoding = encoder.encode_image(test_descriptors, "BoF")
    print(f"BoF编码维度: {bof_encoding.shape}")
    print(f"BoF编码前10个值: {bof_encoding[:10]}")
    
    # 测试VLAD编码
    print("\n测试VLAD编码:")
    vlad_encoding = encoder.encode_image(test_descriptors, "VLAD")
    print(f"VLAD编码维度: {vlad_encoding.shape}")
    print(f"VLAD编码前10个值: {vlad_encoding[:10]}")
    
    # 测试Fisher Vector编码
    print("\n测试Fisher Vector编码:")
    encoder.build_dictionary_gmm(descriptors_list)
    fv_encoding = encoder.encode_image(test_descriptors, "FV")
    print(f"FV编码维度: {fv_encoding.shape}")
    print(f"FV编码前10个值: {fv_encoding[:10]}")

if __name__ == "__main__":
    test_feature_encoding()
