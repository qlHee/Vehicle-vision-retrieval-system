#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取与编码模块
==================
本模块实现图像检索系统的核心算法：
1. FeatureExtractor: 从图像中提取SIFT或ORB局部特征
2. FeatureEncoder: 将局部特征编码为全局图像表示(BoF/VLAD)

工作流程:
  图像 -> 提取局部特征(SIFT/ORB) -> 构建视觉词典(K-means) -> 编码(BoF/VLAD) -> 向量表示
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import time


# =============================================================================
# 特征提取器：从图像中检测关键点并计算描述子
# =============================================================================
class FeatureExtractor:
    """
    特征提取器类
    
    支持两种特征提取算法：
    - SIFT: 尺度不变特征变换，对旋转、缩放、光照变化鲁棒，描述子维度128
    - ORB: 快速二进制特征，计算速度快，描述子维度32
    """
    
    def __init__(self):
        # 创建SIFT检测器（默认参数）
        self.sift = cv2.SIFT_create()
        # 创建ORB检测器，设置最大特征点数为2000以保证足够的特征
        self.orb = cv2.ORB_create(nfeatures=2000)
    
    def extract(self, image, algorithm="SIFT"):
        """
        从图像中提取特征点和描述子
        
        参数:
            image: 输入图像，BGR格式的numpy数组
            algorithm: 特征算法，"SIFT"或"ORB"
        
        返回:
            keypoints: 检测到的关键点列表，包含位置、尺度、方向等信息
            descriptors: 描述子矩阵，形状为(N, 128)或(N, 32)，N为关键点数量
            time_cost: 特征提取耗时（秒）
        """
        start = time.time()
        # 如果是彩色图像，先转换为灰度图（特征提取只需要灰度信息）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # 根据指定算法选择对应的检测器
        detector = self.sift if algorithm == "SIFT" else self.orb
        # 检测关键点并计算描述子
        kp, desc = detector.detectAndCompute(gray, None)
        return kp, desc, time.time() - start


# =============================================================================
# 特征编码器：将局部描述子编码为固定长度的全局向量
# =============================================================================
class FeatureEncoder:
    """
    特征编码器类
    
    将图像的局部特征（多个描述子）编码为单一的全局向量表示，便于图像相似度计算。
    
    支持两种编码方法：
    - BoF (Bag of Features): 视觉词袋模型，统计视觉词出现频率
    - VLAD (Vector of Locally Aggregated Descriptors): 累积残差向量，保留更多细节
    
    可选TF-IDF加权：降低常见视觉词的权重，提高区分性强的视觉词的权重
    """
    
    def __init__(self, codebook_size=256):
        """
        初始化编码器
        
        参数:
            codebook_size: 码本大小（视觉词数量），即K-means聚类中心数
                          值越大表示视觉词典越精细，但计算量也越大
        """
        self.codebook_size = codebook_size
        self.codebook = None      # 视觉词典，存储K个聚类中心，形状(K, 128)
        self.idf_weights = None   # IDF权重向量，形状(K,)，用于TF-IDF加权
    
    # -------------------------------------------------------------------------
    # 步骤1：使用K-means聚类构建视觉词典
    # -------------------------------------------------------------------------
    def build_codebook(self, descriptors_list):
        """
        构建视觉词典（码本）
        
        原理：将所有图像的描述子聚类为K个簇，每个簇中心作为一个"视觉词"
        类似于文本处理中构建词典，但这里是视觉特征的词典
        
        参数:
            descriptors_list: 描述子列表，每个元素是一张图像的描述子矩阵
        
        返回:
            codebook: 视觉词典矩阵，形状(codebook_size, 128)
        """
        # 过滤掉空的描述子，并将所有描述子合并为一个大矩阵
        valid = [d for d in descriptors_list if d is not None and len(d) > 0]
        all_desc = np.vstack(valid).astype(np.float32)
        
        print(f"Building codebook: {len(all_desc)} descriptors -> {self.codebook_size} clusters")
        # 使用K-means聚类，random_state固定保证结果可复现
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10, max_iter=100)
        kmeans.fit(all_desc)
        # 聚类中心即为视觉词典
        self.codebook = kmeans.cluster_centers_.astype(np.float32)
        return self.codebook
    
    # -------------------------------------------------------------------------
    # 步骤2：计算IDF权重（逆文档频率）
    # -------------------------------------------------------------------------
    def compute_idf(self, descriptors_list):
        """
        计算IDF权重
        
        原理：如果某个视觉词在很多图像中都出现，说明它不具有区分性（如背景特征），
              应该降低其权重；反之，只在少数图像中出现的视觉词更有区分性，应提高权重
        
        IDF公式: IDF(w) = log(N / df(w))
        其中N是图像总数，df(w)是包含视觉词w的图像数量
        
        参数:
            descriptors_list: 所有图像的描述子列表
        """
        n_docs = len(descriptors_list)  # 图像总数
        df = np.zeros(self.codebook_size)  # 文档频率：每个视觉词出现在多少张图像中
        
        # 统计每个视觉词的文档频率
        for desc in descriptors_list:
            if desc is None or len(desc) == 0:
                continue
            # 找出该图像中出现了哪些视觉词
            words = self._assign_words(desc)
            # 只统计唯一的视觉词（一张图像中多次出现同一词只计1次）
            df[np.unique(words)] += 1
        
        # 计算IDF权重，加1是为了平滑，避免除零和取log(0)
        self.idf_weights = np.log((n_docs + 1) / (df + 1)) + 1
    
    # -------------------------------------------------------------------------
    # 辅助函数：将描述子分配到最近的视觉词
    # -------------------------------------------------------------------------
    def _assign_words(self, descriptors):
        """
        将每个描述子分配到距离最近的视觉词（硬分配）
        
        参数:
            descriptors: 描述子矩阵，形状(N, 128)
        
        返回:
            words: 视觉词索引数组，形状(N,)，每个元素是0到codebook_size-1的整数
        """
        desc = descriptors.astype(np.float32)
        # 计算每个描述子到每个聚类中心的欧氏距离的平方
        # 使用广播机制：desc[:, np.newaxis, :] 形状变为 (N, 1, 128)
        #              self.codebook[np.newaxis, :, :] 形状变为 (1, K, 128)
        # 相减后形状为 (N, K, 128)，沿最后一维求和得到 (N, K) 的距离矩阵
        diff = desc[:, np.newaxis, :] - self.codebook[np.newaxis, :, :]
        dist = np.sum(diff ** 2, axis=2)
        # 返回每行最小值的索引，即每个描述子对应的最近视觉词
        return np.argmin(dist, axis=1)
    
    # -------------------------------------------------------------------------
    # 步骤3a：BoF编码（词袋模型）
    # -------------------------------------------------------------------------
    def encode_bof(self, descriptors, use_idf=False):
        """
        BoF (Bag of Features) 编码
        
        原理：统计图像中每个视觉词出现的次数，形成直方图
        类似于文本的词袋模型，忽略特征的空间位置关系
        
        流程: 描述子 -> 分配到视觉词 -> 统计词频直方图 -> (可选)TF-IDF加权 -> L2归一化
        
        参数:
            descriptors: 图像的描述子矩阵
            use_idf: 是否使用TF-IDF加权
        
        返回:
            编码向量，形状(codebook_size,)
        """
        # 处理空描述子的情况
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.codebook_size)
        
        # 将每个描述子分配到最近的视觉词
        words = self._assign_words(descriptors)
        # 统计每个视觉词出现的次数（词频TF），bincount统计每个整数值出现的次数
        hist = np.bincount(words, minlength=self.codebook_size).astype(np.float32)
        
        # 如果启用TF-IDF，将词频乘以IDF权重
        if use_idf and self.idf_weights is not None:
            hist = hist * self.idf_weights
        
        # L2归一化，使向量长度为1，便于用余弦相似度比较
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist
    
    # -------------------------------------------------------------------------
    # 步骤3b：VLAD编码（累积残差向量）
    # -------------------------------------------------------------------------
    def encode_vlad(self, descriptors, use_idf=False):
        """
        VLAD (Vector of Locally Aggregated Descriptors) 编码
        
        原理：不仅统计视觉词出现次数，还累积描述子与聚类中心的残差（差向量）
        相比BoF保留了更多的细节信息，通常检索效果更好
        
        流程: 描述子 -> 分配到视觉词 -> 计算残差 -> 按视觉词累加残差 
              -> (可选)IDF加权 -> 展平 -> 幂次归一化 -> L2归一化
        
        参数:
            descriptors: 图像的描述子矩阵
            use_idf: 是否使用IDF加权
        
        返回:
            编码向量，形状(codebook_size * 128,)
        """
        # 处理空描述子的情况
        if descriptors is None or len(descriptors) == 0:
            dim = self.codebook.shape[1] if self.codebook is not None else 128
            return np.zeros(self.codebook_size * dim)
        
        desc = descriptors.astype(np.float32)
        dim = desc.shape[1]  # 描述子维度，SIFT为128
        words = self._assign_words(desc)  # 分配视觉词
        
        # 初始化VLAD矩阵，每个视觉词对应一个128维的累积残差向量
        vlad = np.zeros((self.codebook_size, dim), dtype=np.float32)
        
        # 对每个视觉词，累积所有分配到它的描述子与聚类中心的残差
        for k in range(self.codebook_size):
            mask = (words == k)  # 找出分配到第k个视觉词的描述子
            if np.any(mask):
                # 残差 = 描述子 - 聚类中心，然后求和
                vlad[k] = np.sum(desc[mask] - self.codebook[k], axis=0)
        
        # 如果启用IDF加权，对每个视觉词的残差向量乘以其IDF权重
        if use_idf and self.idf_weights is not None:
            vlad = vlad * self.idf_weights[:, np.newaxis]
        
        # 展平为一维向量
        vlad = vlad.flatten()
        # 幂次归一化（SSR, Signed Square Root）：sign(x) * sqrt(|x|)
        # 这一步可以抑制爆发性视觉词（burstiness）的影响
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        # L2归一化
        norm = np.linalg.norm(vlad)
        return vlad / norm if norm > 0 else vlad
    
    # -------------------------------------------------------------------------
    # 统一编码接口
    # -------------------------------------------------------------------------
    def encode(self, descriptors, method="BoF", use_idf=False):
        """
        统一的编码接口，根据指定方法调用对应的编码函数
        
        参数:
            descriptors: 图像的描述子矩阵
            method: 编码方法，"BoF"或"VLAD"
            use_idf: 是否使用TF-IDF/IDF加权
        
        返回:
            编码向量
        """
        if method == "BoF":
            return self.encode_bof(descriptors, use_idf)
        else:  # VLAD
            return self.encode_vlad(descriptors, use_idf)
