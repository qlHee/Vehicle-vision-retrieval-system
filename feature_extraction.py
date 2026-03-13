#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取和图像匹配系统
实现SIFT、SURF、ORB算法的特征提取和匹配
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Dict, Any

class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self):
        """初始化特征提取器"""
        # 初始化不同的特征检测器
        self.sift = cv2.SIFT_create()
        # SURF需要非免费版本，使用KAZE作为替代
        self.kaze = cv2.KAZE_create()
        self.orb = cv2.ORB_create()
        
        # 初始化匹配器
        self.bf_matcher = cv2.BFMatcher()
        self.flann_matcher = cv2.FlannBasedMatcher()
        
    def extract_sift_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """提取SIFT特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def extract_kaze_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """提取KAZE特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.kaze.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def extract_orb_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """提取ORB特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      algorithm: str = "SIFT") -> List:
        """匹配特征点"""
        if desc1 is None or desc2 is None:
            return []
        
        if algorithm in ["SIFT", "KAZE"]:
            # 使用FLANN匹配器匹配SIFT/KAZE特征
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            if desc1.shape[0] >= 2 and desc2.shape[0] >= 2:
                matches = flann.knnMatch(desc1, desc2, k=2)
                # 应用Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                return good_matches
        else:
            # 使用暴力匹配器匹配ORB特征
            matches = self.bf_matcher.match(desc1, desc2)
            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)
            return matches[:50]  # 返回前50个最佳匹配
        
        return []
    
    def extract_and_time(self, image: np.ndarray, algorithm: str) -> Dict[str, Any]:
        """提取特征并计时"""
        start_time = time.time()
        
        if algorithm == "SIFT":
            keypoints, descriptors = self.extract_sift_features(image)
        elif algorithm == "KAZE":
            keypoints, descriptors = self.extract_kaze_features(image)
        elif algorithm == "ORB":
            keypoints, descriptors = self.extract_orb_features(image)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        end_time = time.time()
        extraction_time = end_time - start_time
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'extraction_time': extraction_time,
            'num_keypoints': len(keypoints) if keypoints else 0
        }

class ImageMatcher:
    """图像匹配器类"""
    
    def __init__(self):
        """初始化图像匹配器"""
        self.feature_extractor = FeatureExtractor()
        self.image_database = {}  # 存储图像特征数据库
        
    def load_image_database(self, image_folder: str):
        """加载图像数据库"""
        print(f"正在加载图像数据库: {image_folder}")
        
        for category_folder in os.listdir(image_folder):
            category_path = os.path.join(image_folder, category_folder)
            if os.path.isdir(category_path) and not category_folder.startswith('.'):
                print(f"处理类别: {category_folder}")
                
                for image_file in os.listdir(category_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(category_path, image_file)
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                # 为每种算法提取特征
                                image_features = {}
                                for algorithm in ["SIFT", "KAZE", "ORB"]:
                                    try:
                                        features = self.feature_extractor.extract_and_time(image, algorithm)
                                        image_features[algorithm] = features
                                    except Exception as e:
                                        print(f"提取{algorithm}特征失败: {e}")
                                        image_features[algorithm] = None
                                
                                self.image_database[image_path] = {
                                    'image': image,
                                    'category': category_folder,
                                    'features': image_features
                                }
                        except Exception as e:
                            print(f"加载图像失败 {image_path}: {e}")
        
        print(f"成功加载 {len(self.image_database)} 张图像")
    
    def find_best_match(self, query_image: np.ndarray, algorithm: str = "SIFT") -> Dict[str, Any]:
        """找到最佳匹配图像"""
        if not self.image_database:
            return None
        
        # 提取查询图像特征
        query_features = self.feature_extractor.extract_and_time(query_image, algorithm)
        
        if query_features['descriptors'] is None:
            return None
        
        best_match = None
        best_score = 0
        match_results = []
        
        start_time = time.time()
        
        for image_path, image_data in self.image_database.items():
            if image_data['features'][algorithm] is None:
                continue
                
            db_features = image_data['features'][algorithm]
            if db_features['descriptors'] is None:
                continue
            
            # 匹配特征
            matches = self.feature_extractor.match_features(
                query_features['descriptors'], 
                db_features['descriptors'], 
                algorithm
            )
            
            # 计算匹配分数
            num_matches = len(matches)
            if num_matches > best_score:
                best_score = num_matches
                best_match = {
                    'image_path': image_path,
                    'image': image_data['image'],
                    'category': image_data['category'],
                    'matches': matches,
                    'num_matches': num_matches,
                    'query_keypoints': query_features['keypoints'],
                    'db_keypoints': db_features['keypoints']
                }
            
            match_results.append({
                'image_path': image_path,
                'category': image_data['category'],
                'num_matches': num_matches
            })
        
        end_time = time.time()
        matching_time = end_time - start_time
        
        if best_match:
            best_match['matching_time'] = matching_time
            best_match['query_extraction_time'] = query_features['extraction_time']
            best_match['query_keypoints_num'] = query_features['num_keypoints']
            best_match['all_results'] = sorted(match_results, key=lambda x: x['num_matches'], reverse=True)
        
        return best_match

def test_feature_extraction():
    """测试特征提取功能"""
    print("测试特征提取功能...")
    
    # 创建图像匹配器
    matcher = ImageMatcher()
    
    # 加载图像数据库
    image_folder = "/Users/tuxol/Documents/DataVault/CST/Visual Computing/Lab1/image"
    matcher.load_image_database(image_folder)
    
    # 测试匹配
    if matcher.image_database:
        # 随机选择一张图像作为查询图像
        test_image_path = list(matcher.image_database.keys())[0]
        test_image = matcher.image_database[test_image_path]['image']
        
        print(f"\n使用测试图像: {test_image_path}")
        
        # 测试不同算法
        for algorithm in ["SIFT", "KAZE", "ORB"]:
            print(f"\n测试 {algorithm} 算法:")
            try:
                result = matcher.find_best_match(test_image, algorithm)
                if result:
                    print(f"  最佳匹配: {result['image_path']}")
                    print(f"  匹配点数: {result['num_matches']}")
                    print(f"  查询特征提取时间: {result['query_extraction_time']:.4f}秒")
                    print(f"  匹配时间: {result['matching_time']:.4f}秒")
                    print(f"  查询图像关键点数: {result['query_keypoints_num']}")
                else:
                    print(f"  {algorithm} 算法匹配失败")
            except Exception as e:
                print(f"  {algorithm} 算法测试失败: {e}")

if __name__ == "__main__":
    test_feature_extraction()
