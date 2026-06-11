#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆视觉检索系统 - 结果重排序(Re-ranking)模块
=============================================
本模块从 gui_app.py 中分离出"检索结果重排序"相关逻辑，以 Mixin 形式提供给
ImageSearchApp 继承使用。分离后行为与原实现完全一致。

包含两种重排序方法：
  1. 线性组合重排序(_reorder_results_linear)：
     提取颜色(HSV直方图)、纹理(梯度方向直方图)、形状(Hu矩+边缘方向)特征，
     与原始检索得分做加权线性组合后重新排序。
  2. 基于图的重排序(_reorder_results_graph / _graph_based_rerank)：
     构建结果间相似性图，使用 manifold ranking 做信息传播，
     再与原始得分融合后重新排序。

以及共用的底层特征/相似度计算与重排序结果展示：
  - _extract_color_hist / _extract_texture_lbp / _extract_shape_feature
  - _compute_similarity
  - _show_reordered_results

说明：
  - RerankingMixin 仅承载方法，不持有独立状态；所有 self.* 属性
    (extractor / encoder / orb_encoder / database_data / current_results /
     query_image / query_path / result_canvases / info_text / evaluator /
     pr_ax / pr_figure / pr_canvas / status_var 等)均由 ImageSearchApp 提供。
  - 依赖的 self._display_image 等方法仍定义在 ImageSearchApp 中，通过继承解析。
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox


class RerankingMixin:
    """检索结果重排序功能 Mixin，供 ImageSearchApp 继承。"""

    # =========================================================================
    # 底层特征与相似度计算（线性组合重排序使用）
    # =========================================================================
    def _extract_color_hist(self, image):
        """
        提取颜色特征：HSV 颜色直方图

        思路：把图像从 BGR 转到 HSV 颜色空间，分别统计色调(H)、饱和度(S)、
        明度(V)三个通道的颜色分布直方图，拼成一个向量来描述"这张图大概是
        什么颜色构成的"。颜色相近的车通常这个向量也相近。

        参数:
            image: BGR 格式的彩色图像(numpy 数组)
        返回:
            归一化后的颜色特征向量，长度 = 32+32+32 = 96
        """
        # OpenCV 读进来的图是 BGR 顺序，先转成 HSV。HSV 比 RGB 更接近人对
        # 颜色的直觉(色调/鲜艳程度/明暗)，做颜色统计更稳定。
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 分别对 H/S/V 三个通道统计直方图，每个通道分成 32 个区间(bin)。
        # 注意 H(色调)的取值范围是 0~180(OpenCV 的约定)，S/V 是 0~256。
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        # 把三个通道的直方图首尾拼接，再压成一维向量
        hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        # L2 归一化(让向量长度变成 1)，这样不同图像之间才好用余弦相似度比较，
        # 不会因为图片大小/像素多少而影响结果。norm 为 0(全黑图)时直接返回原值避免除零。
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _extract_texture_lbp(self, image):
        """
        提取纹理特征：梯度方向直方图

        思路：纹理 = 图像里明暗变化的方式。用 Sobel 算子求出每个像素在
        水平/垂直方向的亮度变化(梯度)，由此算出"变化的方向"和"变化的强度"，
        再统计各个方向上的强度分布。纹理相似的图，这个方向分布也相似。

        参数:
            image: 彩色或灰度图像
        返回:
            归一化后的纹理特征向量，长度 = 36(把 0~360 度分成 36 份)
        """
        # 纹理只看明暗变化，不需要颜色，所以先转灰度图(若本身是灰度则跳过)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Sobel 算子：sobelx 是水平方向的亮度变化，sobely 是垂直方向的亮度变化
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # 梯度强度(变化有多剧烈) = 水平和垂直变化的平方和开根号
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 梯度方向(变化朝哪个方向) = arctan2(y, x)，转成角度并平移到 0~360 度范围
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi + 180
        # 把所有像素按"方向"分到 36 个区间，并用"强度"作为权重累加
        # (方向相同但变化越剧烈的像素，贡献越大)
        hist, _ = np.histogram(angle.flatten(), bins=36, range=(0, 360), weights=magnitude.flatten())
        # L2 归一化
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _extract_shape_feature(self, image):
        """
        提取形状特征：Hu 矩 + 边缘方向直方图

        思路：用两类信息描述"形状"——
          1) Hu 矩：7 个数字，对图像缩放/旋转/平移都不敏感，是经典的形状描述子；
          2) 边缘方向直方图：先用 Canny 找出物体轮廓边缘，再统计边缘走向的分布。
        两者拼起来共同刻画轮廓形状。

        参数:
            image: 彩色或灰度图像
        返回:
            归一化后的形状特征向量，长度 = 7(Hu 矩) + 18(边缘方向) = 25
        """
        # 形状只看轮廓，不需要颜色，先转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # 计算图像矩，再由矩推出 7 个 Hu 不变矩(对平移/旋转/缩放稳定)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Hu 矩数值跨度极大(从很大到接近 0)，这里做对数压缩并保留正负号，
        # 让 7 个数字落在差不多的量级，便于后续比较。加 1e-10 防止 log10(0)。
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        # Canny 边缘检测：找出图像中明显的轮廓线(返回黑白边缘图)
        edges = cv2.Canny(gray, 100, 200)
        # 在边缘图上再求一次梯度，得到每条边缘的走向(方向)
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi + 180
        # 把边缘方向分到 18 个区间，统计每个方向上有多少边缘(这次不加权重)
        edge_hist, _ = np.histogram(angle.flatten(), bins=18, range=(0, 360))
        # 把 Hu 矩(7 维)和边缘方向直方图(18 维)拼成一个形状向量
        feature = np.concatenate([hu_moments, edge_hist.astype(np.float64)])
        # L2 归一化
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature

    def _compute_similarity(self, feat1, feat2):
        """
        计算两个特征向量的余弦相似度

        余弦相似度 = 两个向量夹角的余弦值 = 点积 / (模长1 × 模长2)。
        值越接近 1 表示两个特征越像，越接近 0 表示越不相关。

        参数:
            feat1, feat2: 两个等长的特征向量
        返回:
            相似度，范围约 [0, 1]
        """
        # 分别计算两个向量的模长(长度)
        norm1, norm2 = np.linalg.norm(feat1), np.linalg.norm(feat2)
        # 任一向量为零向量时无法计算夹角，直接返回 0(不相关)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 点积除以两个模长的乘积，即为余弦相似度
        return np.dot(feat1, feat2) / (norm1 * norm2)

    # =========================================================================
    # 基于线性组合的图像重排序
    # =========================================================================
    def _reorder_results_linear(self):
        """
        基于线性组合的图像重排序

        复用“车辆检索系统-基于线性组合的重排序”的实现思路：
        1. 提取查询图像和候选图像的颜色、纹理、形状特征
        2. 计算各特征与原始检索结果的相似度
        3. 将原始检索得分与各特征相似度进行线性组合
        4. 根据组合得分重新排序
        """
        if self.current_results is None or len(self.current_results) == 0:
            messagebox.showwarning("Warning", "Please run search first")
            return

        if self.query_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.status_var.set("Reordering (Linear)...")

        # 第1步：提取"查询图像"的颜色、纹理、形状三种特征(后面拿它和每个候选比)
        query_color = self._extract_color_hist(self.query_image)
        query_texture = self._extract_texture_lbp(self.query_image)
        query_shape = self._extract_shape_feature(self.query_image)

        # 各项权重：原始检索得分占大头(0.7)，颜色/纹理/形状作为补充微调。
        # 四个权重加起来为 1。可以理解为"主要信原来的排序，再用低层特征小幅修正"。
        w_original, w_color, w_texture, w_shape = 0.7, 0.15, 0.08, 0.07

        # 原始检索结果里 distance 越小越相似。下面要把 distance 反转/归一化成
        # "越大越相似"的得分，所以先求出当前这批结果里 distance 的最大最小值。
        max_dist = max(r['distance'] for r in self.current_results)
        min_dist = min(r['distance'] for r in self.current_results)
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0

        # 第2步：逐个处理每个候选结果，算出它的综合得分
        reordered = []
        for r in self.current_results:
            img = cv2.imread(r['path'])
            if img is None:
                continue  # 图片读不出来就跳过

            # 把原始 distance 映射成 [0,1] 的相似度：distance 最小的→1，最大的→0
            orig_sim = 1 - (r['distance'] - min_dist) / dist_range if dist_range > 0 else 1.0
            # 提取该候选图的三种低层特征
            res_color = self._extract_color_hist(img)
            res_texture = self._extract_texture_lbp(img)
            res_shape = self._extract_shape_feature(img)

            # 分别计算候选图与查询图在颜色/纹理/形状上的相似度
            sim_color = self._compute_similarity(query_color, res_color)
            sim_texture = self._compute_similarity(query_texture, res_texture)
            sim_shape = self._compute_similarity(query_shape, res_shape)

            # 第3步：把"原始得分"和三种特征相似度按权重线性相加，得到综合得分
            combined_score = (
                w_original * orig_sim +
                w_color * sim_color +
                w_texture * sim_texture +
                w_shape * sim_shape
            )

            # 保存这个候选的综合得分及各分项(分项用于在信息栏展示，便于调试)
            reordered.append({
                'path': r['path'],
                'label': r['label'],
                'distance': 1 - combined_score,  # 转回 distance 形式(越小越好)以兼容显示逻辑
                'combined_score': combined_score,
                'orig_sim': orig_sim,
                'sim_color': sim_color,
                'sim_texture': sim_texture,
                'sim_shape': sim_shape,
            })

        # 第4步：按综合得分从高到低重新排序，得分高的排前面
        reordered.sort(key=lambda x: x['combined_score'], reverse=True)
        self.current_results = reordered
        self._show_reordered_results(reordered, method_name="Linear Combination", detail_line="Orig(0.7) + Color(0.15) + Tex(0.08) + Shape(0.07)")
        self.status_var.set("Linear Reordering Done")

    # =========================================================================
    # 基于图的图像重排序
    # =========================================================================
    def _reorder_results_graph(self):
        """
        使用基于图的方法对检索结果进行重排序

        基于图的重排序步骤：
        1. 构建相似性图：节点为图像，边权重为图像间相似度
        2. 多图学习：结合查询图像与检索结果之间的关系
        3. 图排序：使用manifold ranking算法重新排序
        """
        if self.current_results is None or len(self.current_results) == 0:
            messagebox.showwarning("Warning", "Please run search first")
            return

        self.status_var.set("Reordering (Graph)...")

        # 读取当前界面上选择的算法/编码方式，确定用哪个编码器和描述子字段
        algo = self.algorithm_var.get()
        method = self.encoding_var.get()
        use_idf = self.use_idf_var.get()

        encoder = self.encoder if algo == "SIFT" else self.orb_encoder
        desc_key = 'sift_desc' if algo == "SIFT" else 'orb_desc'

        # 把查询图像编码成向量(图重排序要用它和候选向量算相似度)
        _, query_desc, _ = self.extractor.extract(self.query_image, algo)
        query_enc = encoder.encode(query_desc, method, use_idf)

        # 把每个候选结果也编码成向量。描述子直接从预处理缓存里取，省去重复提特征。
        result_encodings = []
        valid_results = []
        for r in self.current_results:
            desc = self.database_data.get(r['path'], {}).get(desc_key)
            if desc is not None:
                enc = encoder.encode(desc, method, use_idf)
                result_encodings.append(enc)
                valid_results.append(r)

        # 图重排序需要候选之间互相比较，至少要有 2 个候选才有意义
        if len(valid_results) < 2:
            self.status_var.set("Not enough results to reorder")
            return

        # 调用核心算法做图传播重排序，再展示结果
        reordered_results = self._graph_based_rerank(query_enc, result_encodings, valid_results)
        self.current_results = reordered_results

        self._show_reordered_results(reordered_results, method_name="Graph-based", detail_line="Manifold Ranking + Original Score")
        self.status_var.set("Graph Reordering Done")

    def _graph_based_rerank(self, query_enc, result_encodings, results):
        """
        基于图的重排序算法实现（流形排序 / Manifold Ranking）

        通俗理解：把每个候选结果看成"图"上的一个点，点和点之间用"有多像"
        连一条边。查询图像给每个点一个初始分数(和查询越像分越高)，然后让分数
        沿着边在图上反复传播——如果一个点周围都是高分点，它自己的分也会被
        "带高"。这样能利用候选之间的群聚关系，把真正同类的图一起往前提。

        算法步骤：
        1. 构建相似性图：计算所有候选图像之间的相似度矩阵
        2. 图内学习：利用候选之间的相似关系反复传播分数(迭代收敛)
        3. 图间学习：把查询图像与候选的相似度作为传播的初始/锚定分数
        4. 融合排序：综合"传播后的分数"和"原始检索分数"得到最终排序

        参数:
            query_enc: 查询图像的编码向量
            result_encodings: 各候选图像的编码向量列表
            results: 与 result_encodings 对应的原始检索结果列表
        返回:
            按最终得分重新排序后的结果列表
        """
        # ---- 准备：把所有候选向量做 L2 归一化，方便用点积直接当余弦相似度 ----
        encodings = np.array(result_encodings)
        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 防止除以 0
        normalized_encodings = encodings / norms

        # ---- 第1步：构建相似度矩阵 ----
        # similarity_matrix[i][j] = 候选 i 和候选 j 的余弦相似度(归一化后点积)
        similarity_matrix = np.dot(normalized_encodings, normalized_encodings.T)
        # 查询向量也归一化，算出"每个候选与查询"的相似度，作为后面的初始分数 y
        query_norm = np.linalg.norm(query_enc)
        normalized_query = query_enc / query_norm if query_norm > 0 else query_enc
        query_similarity = np.dot(normalized_encodings, normalized_query)

        # ---- 第2步：把相似度转成图的"亲和度"(边权重) ----
        # 用高斯核把相似度映射成边权重：越相似权重越接近 1，越不像越接近 0。
        # sigma 控制衰减快慢。然后把对角线(自己到自己)置 0，不让自环影响传播。
        sigma = 0.5
        affinity_matrix = np.exp((similarity_matrix - 1) / (2 * sigma ** 2))
        np.fill_diagonal(affinity_matrix, 0)

        # ---- 对称归一化邻接矩阵(流形排序的标准做法，保证迭代稳定收敛) ----
        # degree[i] = 第 i 个点所有边权重之和(它在图里的"连接强度")
        degree = np.sum(affinity_matrix, axis=1)
        degree[degree == 0] = 1
        # S = D^(-1/2) · A · D^(-1/2)，用度数对边权重做归一化
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        normalized_affinity = D_inv_sqrt @ affinity_matrix @ D_inv_sqrt

        # ---- 第3步：迭代传播分数(流形排序的核心公式) ----
        # f = α·(S·f) + (1-α)·y
        #   y 是初始分数(每个候选与查询的相似度)，相当于"锚"，每轮都拉回一点；
        #   S·f 是从邻居那里传播过来的分数；
        #   α 控制"听邻居的"和"守住初始分"的比例。反复迭代直到分数基本不变。
        alpha = 0.5
        y = query_similarity.copy()
        f = y.copy()
        for _ in range(20):  # 最多迭代 20 次
            f_new = alpha * (normalized_affinity @ f) + (1 - alpha) * y
            # 如果这一轮和上一轮几乎一样，说明已经收敛，提前结束
            if np.allclose(f, f_new, atol=1e-6):
                break
            f = f_new

        # ---- 第4步：把"传播后分数 f"和"原始检索分数"融合 ----
        # 原始 distance 越小越好，这里转成"越大越好"的分数
        original_scores = np.array([1.0 / (1.0 + r['distance']) for r in results])

        # 两组分数量纲不同，各自归一化到 [0,1] 再融合才公平
        f_normalized = (f - f.min()) / (f.max() - f.min()) if f.max() > f.min() else f
        orig_normalized = (
            (original_scores - original_scores.min()) / (original_scores.max() - original_scores.min())
            if original_scores.max() > original_scores.min() else original_scores
        )

        # beta 控制两者比重：0.6 偏向图传播结果，0.4 保留原始排序
        beta = 0.6
        final_scores = beta * f_normalized + (1 - beta) * orig_normalized
        # argsort 默认升序，加负号变降序，得到"分数从高到低"的下标顺序
        sorted_indices = np.argsort(-final_scores)

        # 按新顺序重组结果列表，并把最终分数写回 distance 字段(越小越好)以兼容显示
        reordered_results = []
        for idx in sorted_indices:
            result = results[idx].copy()
            result['distance'] = float(1.0 - final_scores[idx])
            reordered_results.append(result)

        return reordered_results

    # =========================================================================
    # 重排序结果展示
    # =========================================================================
    def _show_reordered_results(self, results, method_name, detail_line):
        """
        显示重排序后的结果

        把重排序后的 Top10 图像重新画到界面上(绿框=同类正确，红框=不同类错误)，
        在信息栏写出方法说明和各项得分，并重绘 PR 曲线。

        参数:
            results: 重排序后的结果列表
            method_name: 方法名(显示用，如 "Linear Combination")
            detail_line: 方法细节说明(显示用，如各项权重)
        """
        # 从查询图像路径推断它的真实标签(用所在文件夹名当车牌号)
        query_label = os.path.basename(os.path.dirname(self.query_path))

        # 先把 10 个结果槽位全部清空(图像、文字、勾选框)
        for i in range(len(self.result_canvases)):
            self.result_canvases[i].delete("all")
            self.result_label_vars[i].set("")
            self.result_label_widgets[i].config(foreground="black")
            self.result_vars[i].set(False)

        # 逐个把重排序后的结果画到对应槽位上
        for i, canvas in enumerate(self.result_canvases):
            if i < len(results):
                r = results[i]
                # 标签和查询相同→绿色(正确)，不同→红色(错误)
                color = "green" if r['label'] == query_label else "red"
                self.result_label_vars[i].set(f"{i+1}. {r['label']}")
                self.result_label_widgets[i].config(foreground=color)
                self._display_image(cv2.imread(r['path']), canvas)
                canvas.update()
                # 给图片画一圈彩色边框，直观标注对错
                canvas.create_rectangle(1, 1, canvas.winfo_width() - 1, canvas.winfo_height() - 1,
                                        outline=color, width=3)

        # 计算 Top10 精度(前 10 个结果里有多少个标签和查询相同)
        retrieved = [r['label'] for r in results]
        precision = sum(1 for l in retrieved if l == query_label) / len(retrieved) if retrieved else 0

        # 在信息栏拼出文字报告
        algo, method, use_idf = self.algorithm_var.get(), self.encoding_var.get(), self.use_idf_var.get()
        idf_str = " (TF-IDF)" if use_idf else ""
        info = f"=== Reordered Results ===\n"
        info += f"Method: {algo} + {method}{idf_str} + {method_name}\n"
        info += f"Rerank: {detail_line}\n"
        info += f"Query: {os.path.basename(self.query_path)}\n"
        info += f"True Label: {query_label}\n\n"
        info += f"Top10 Precision: {precision*100:.1f}%\n\n"
        info += "=== Top10 Results ===\n"
        for i, r in enumerate(results[:10]):
            mark = "Y" if r['label'] == query_label else "N"  # Y=正确 N=错误
            # 线性重排序有 combined_score 字段，图重排序只有 distance，按需显示
            score_key = 'combined_score' if 'combined_score' in r else 'distance'
            info += f"{i+1}. [{mark}] {r['label']} ({score_key}:{r[score_key]:.4f})\n"
            # 线性重排序额外显示颜色/纹理/形状三项分相似度
            if 'sim_color' in r:
                info += f"    Color:{r['sim_color']:.3f} Tex:{r['sim_texture']:.3f} Shape:{r['sim_shape']:.3f}\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)

        # ---- 重绘单次查询的 PR 曲线(精度-召回率曲线) ----
        labels = [r['label'] for r in results]
        # 该类别在数据库中的总数，作为召回率的分母
        total_rel = self.evaluator.label_counts.get(query_label, 1)
        precs, recs = [], []
        # 对 K=1..N 逐点计算精度和召回率
        for k in range(1, len(labels) + 1):
            rel = sum(1 for l in labels[:k] if l == query_label)  # 前 k 个里的正确数
            precs.append(rel / k)                                  # Precision@k
            recs.append(rel / total_rel if total_rel > 0 else 0)   # Recall@k
        self.pr_ax.clear()
        self.pr_ax.plot(recs, precs, 'r-o', linewidth=2, markersize=4)
        self.pr_ax.set_xlabel('Recall')
        self.pr_ax.set_ylabel('Precision')
        self.pr_ax.set_title(f'PR Curve ({query_label})')
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1)
        self.pr_ax.grid(True, alpha=0.3)
        self.pr_figure.tight_layout()
        self.pr_canvas.draw()
