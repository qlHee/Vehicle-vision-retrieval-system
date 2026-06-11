#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆视觉检索系统 - 查询扩展(Query Expansion)模块
================================================
本模块从 gui_app.py 中分离出"查询扩展"相关逻辑，以 Mixin 形式提供给
ImageSearchApp 继承使用。分离后行为与原实现完全一致：

查询扩展(Query Expansion, QE)思路：
  1. 取查询图像本身的编码向量，再取若干高置信度检索结果(用户勾选项，
     或 Top-K)的编码向量；
  2. 对这些向量求均值并归一化，得到一个"扩展后的查询向量"；
  3. 用扩展向量重新检索，从而召回更多同类目标。

说明：
  - QueryExpansionMixin 仅承载方法，不持有任何独立状态；所有 self.* 属性
    (extractor / encoder / orb_encoder / database_data / index_cache /
     current_results / result_vars / status_var 等)均由 ImageSearchApp 提供。
  - 依赖的 self._get_query_encoding / self._search_with_encoding /
    self._show_results 等方法仍定义在 ImageSearchApp 中，通过继承解析。
"""

import os
import numpy as np
from tkinter import messagebox


class QueryExpansionMixin:
    """查询扩展功能 Mixin，供 ImageSearchApp 继承。"""

    def _expand_query_encoding(self, query_enc, results, algo, method, use_idf):
        """
        基于查询图像和 Top5 结果执行 Query Expansion(查询扩展)。

        核心思想：单张查询图像可能拍得不够典型(角度/光照偏)，于是把它和
        几个最像的检索结果"取平均"，得到一个更有代表性的查询向量，再用它
        去检索，往往能召回更多同类目标。

        参数:
            query_enc: 查询图像的编码向量
            results: 检索结果列表(取前 5 个参与扩展)
            algo/method/use_idf: 当前的特征算法/编码方法/是否用 IDF
        返回:
            扩展并归一化后的查询向量；若没有可用结果则原样返回 query_enc
        """
        if not results:
            return query_enc

        # 根据算法选择对应的编码器和描述子字段
        encoder = self.encoder if algo == "SIFT" else self.orb_encoder
        desc_key = 'sift_desc' if algo == "SIFT" else 'orb_desc'
        # candidates 收集要参与平均的向量，第一个就是查询本身
        candidates = [query_enc]

        # 取前 5 个检索结果，把它们的编码向量也加进来
        for r in results[:5]:
            desc = self.database_data.get(r['path'], {}).get(desc_key)
            if desc is None:
                continue
            candidates.append(encoder.encode(desc, method, use_idf))

        # 如果一个候选都没加进来(只有查询自己)，无法扩展，原样返回
        if len(candidates) == 1:
            return query_enc

        # 对所有向量按列求平均，得到"扩展查询向量"，再 L2 归一化
        expanded = np.mean(np.vstack(candidates), axis=0)
        norm = np.linalg.norm(expanded)
        return expanded / norm if norm > 0 else expanded

    def _query_expansion(self):
        """
        使用"用户勾选的结果"或"Top-K 结果"对查询向量做 Query Expansion 并重新检索。

        流程：
          1. 取查询图像的编码向量；
          2. 收集勾选的结果(若没勾选则取前 K 个)的编码向量；
          3. 把它们和查询向量一起求平均、归一化，得到扩展查询向量；
          4. 用扩展向量重新检索，刷新界面结果。
        """
        # 前置检查：数据库已加载、已选图、已检索过
        if not self.database_loaded or self.query_image is None:
            messagebox.showwarning("Warning", "Please load database and select image first")
            return
        if not self.current_results:
            messagebox.showwarning("Warning", "Please run search first")
            return

        # 读取当前界面选择的算法/编码/IDF
        algo, method, use_idf = self.algorithm_var.get(), self.encoding_var.get(), self.use_idf_var.get()
        self.status_var.set("Query Expansion...")

        # 第1步：取查询图像编码向量(ext_time 是特征提取耗时，显示用)
        query_enc, ext_time = self._get_query_encoding(algo, method, use_idf)
        encoder = self.encoder if algo == "SIFT" else self.orb_encoder
        desc_key = 'sift_desc' if algo == "SIFT" else 'orb_desc'

        # 第2步：找出用户勾选了哪些结果(result_vars 里被勾选的下标)
        selected_indices = [i for i, var in enumerate(self.result_vars[:len(self.current_results)]) if var.get()]
        candidates = [query_enc]  # 参与平均的向量，查询自己先放进去

        # 用户勾选了就用勾选的；没勾选就用前 qe_k_var 个结果(界面上的下拉框)
        if selected_indices:
            selected_results = [self.current_results[i] for i in selected_indices]
            source_results = selected_results
        else:
            source_results = self.current_results[:self.qe_k_var.get()]

        # 把这些参考结果的编码向量都加进 candidates
        for r in source_results:
            desc = self.database_data.get(r['path'], {}).get(desc_key)
            if desc is None:
                continue
            candidates.append(encoder.encode(desc, method, use_idf))

        # 第3步：求平均得到扩展查询向量，并 L2 归一化
        expanded_query = np.mean(np.vstack(candidates), axis=0)
        norm = np.linalg.norm(expanded_query)
        if norm > 0:
            expanded_query = expanded_query / norm

        # 第4步：用扩展向量重新检索(排除查询图本身)，刷新显示
        results, search_time = self._search_with_encoding(expanded_query, algo, method, use_idf, exclude_path=self.query_path, k=10)

        self.current_results = results
        self.query_expanded = True  # 标记已扩展
        self._show_results(results, ext_time, search_time, algo, method, use_idf)
        # 重置所有勾选框，方便下一次操作
        for var in self.result_vars:
            var.set(False)
        self.status_var.set("Query Expansion Done")
