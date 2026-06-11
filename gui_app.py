#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆视觉检索系统 - GUI界面模块
==============================
本模块实现图像检索系统的图形用户界面，提供以下功能：
1. 选择查询图像并显示
2. 切换特征提取算法（SIFT/ORB）和编码方法（BoF/VLAD）
3. 启用/禁用TF-IDF加权
4. 执行图像检索并显示Top10结果
5. 运行测试集评估并显示性能指标
6. 显示PR曲线和特征编码直方图

界面布局:
  +------------------+---------------+-------------+------------+
  | 查询图像          | 控制面板       | 检索信息    | PR曲线      |
  +------------------+---------------+-------------+------------+
  |                  检索结果 (Top 10)                           |
  +-------------------------------------------------------------+
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，使matplotlib可以嵌入Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from feature_extractor import FeatureExtractor, FeatureEncoder
from image_retrieval import ImageRetriever, PerformanceEvaluator, load_image_database
from reranking import RerankingMixin
from query_expansion import QueryExpansionMixin


# 缓存格式版本号。version>=2 使用"相对路径"存储，保证项目被复制/移动后缓存依然可用
CACHE_FORMAT_VERSION = 2


def compute_database_hash(image_folder):
    """
    计算数据库文件夹的哈希值，用于检测数据库是否发生变化。

    关键点：使用相对于image_folder的"相对路径"+文件大小生成哈希，
    并对所有条目排序后再计算，因此：
      - 项目被复制或移动到其他绝对路径后，哈希值保持不变（缓存仍可用）；
      - 不同操作系统/文件系统的目录遍历顺序差异也不会影响结果。
    （沿用原设计：只用文件大小而非修改时间，避免无意义的误判。）

    参数:
        image_folder: 数据库图像根目录

    返回:
        md5哈希字符串
    """
    hash_data = []
    for root, dirs, files in os.walk(image_folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]  # 跳过隐藏文件夹
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(root, f)
                fsize = os.path.getsize(filepath)
                relpath = os.path.relpath(filepath, image_folder)
                hash_data.append(f"{relpath}:{fsize}")
    hash_data.sort()  # 排序保证跨运行/跨平台的确定性
    return hashlib.md5('\n'.join(hash_data).encode()).hexdigest()


class ImageSearchApp(RerankingMixin, QueryExpansionMixin):
    """
    图像检索系统GUI应用主类

    该类整合了所有GUI组件和检索功能，实现完整的交互式图像检索系统。
    启动时会异步加载数据库并预计算所有编码组合，以实现快速响应。

    重排序(Re-ranking)逻辑由 reranking.RerankingMixin 提供，
    查询扩展(Query Expansion)逻辑由 query_expansion.QueryExpansionMixin 提供，
    二者均通过继承混入，行为与分离前完全一致。
    """
    
    def __init__(self, root, image_folder, test_folder):
        """
        初始化应用
        
        参数:
            root: Tkinter根窗口对象
            image_folder: 数据库图像文件夹路径（包含多个车牌号子文件夹）
            test_folder: 测试集图像文件夹路径
        """
        self.root = root
        self.root.title("Vehicle Image Retrieval System")
        self.root.geometry("1400x900")
        
        # 路径配置
        self.image_folder = image_folder  # 数据库图像目录
        self.test_folder = test_folder    # 测试集目录
        
        # 核心组件初始化
        self.extractor = FeatureExtractor()           # 特征提取器
        self.encoder = FeatureEncoder(codebook_size=256)       # SIFT编码器，256个视觉词
        self.orb_encoder = FeatureEncoder(codebook_size=256)   # ORB编码器（单独的codebook）
        self.retriever = ImageRetriever(metric="cosine")  # 图像检索器
        self.evaluator = None                          # 性能评估器（加载数据库后初始化）
        
        # 状态变量
        self.database_loaded = False   # 数据库是否加载完成
        self.query_image = None        # 当前查询图像（numpy数组）
        self.query_path = None         # 当前查询图像路径
        self.current_results = None    # 当前检索结果列表
        self.query_expanded = False    # 是否已执行Query Expansion
        self.database_data = {}        # 数据库数据缓存 {路径: {label, sift_desc, orb_desc}}
        self.index_cache = {}          # 预计算的检索索引缓存 {(算法, 编码, IDF): retriever}
        self.linear_feature_cache = {}  # 线性重排序缓存 {path: {color, texture, shape}}
        
        # 缓存文件路径
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '.cache')
        self.cache_file = os.path.join(self.cache_dir, 'database_cache.pkl')
        
        # GUI控件变量（用于获取用户选择）
        self.algorithm_var = tk.StringVar(value="SIFT")   # 特征算法选择
        self.encoding_var = tk.StringVar(value="BoF")     # 编码方法选择
        self.use_idf_var = tk.BooleanVar(value=False)     # 是否使用TF-IDF
        
        # 创建界面并异步加载数据库
        self._create_widgets()
        self._load_database_async()
    
    # =========================================================================
    # 界面布局构建
    # =========================================================================
    def _create_widgets(self):
        """
        创建GUI界面布局
        
        界面分为上下两部分：
        - 上部：查询图像 | 控制面板 | 检索信息 | PR曲线
        - 下部：检索结果展示（2行5列，共10张图）
        """
        # 主框架，包含所有控件
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # ----- 上部区域：固定高度380像素 -----
        top = ttk.Frame(main, height=380)
        top.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        top.pack_propagate(False)  # 禁止子控件改变父容器大小
        
        # 查询图像显示区域（左侧）
        qf = ttk.LabelFrame(top, text="Query Image", padding=5, width=362)
        qf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        qf.pack_propagate(False)
        self.query_canvas = tk.Canvas(qf, bg="white")  # 用Canvas显示图像
        self.query_canvas.pack(fill=tk.BOTH, expand=True)

        # 控制面板（中左）- 包含算法选择、按钮等
        ctrl = ttk.LabelFrame(top, text="Control Panel", padding=20, width=300)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ctrl.pack_propagate(False)
        
        # 特征算法双选框：默认SIFT，勾选ORB后SIFT自动取消
        ttk.Label(ctrl, text="Feature:").grid(row=0, column=0, sticky=tk.W, pady=(0, 2))
        feature_row = ttk.Frame(ctrl)
        feature_row.grid(row=0, column=1, sticky=tk.W, pady=(0, 2))
        self.feature_sift_var = tk.BooleanVar(value=True)
        self.feature_orb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(feature_row, text="SIFT", variable=self.feature_sift_var,
                        command=self._on_feature_toggle).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(feature_row, text="ORB", variable=self.feature_orb_var,
                        command=self._on_feature_toggle).pack(side=tk.LEFT)
        # 编码方法双选框：默认BoF，勾选VLAD后BoF自动取消
        ttk.Label(ctrl, text="Encoding:").grid(row=1, column=0, sticky=tk.W, pady=2)
        encoding_row = ttk.Frame(ctrl)
        encoding_row.grid(row=1, column=1, sticky=tk.W, pady=2)
        self.encoding_bof_var = tk.BooleanVar(value=True)
        self.encoding_vlad_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(encoding_row, text="BoF ", variable=self.encoding_bof_var,
                        command=self._on_encoding_toggle).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(encoding_row, text="VLAD", variable=self.encoding_vlad_var,
                        command=self._on_encoding_toggle).pack(side=tk.LEFT)
        # TF-IDF开关复选框 + Evaluate按钮同一行
        tfidf_eval_row = ttk.Frame(ctrl)
        tfidf_eval_row.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=1)
        tfidf_eval_row.columnconfigure(0, weight=1)
        tfidf_eval_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(tfidf_eval_row, text="Enable TF-IDF", variable=self.use_idf_var
                       ).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(tfidf_eval_row, text="Evaluate", command=self._run_evaluation
                  ).grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))
        
        # 分割线：
        ttk.Separator(ctrl, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=8)
        
        # 功能按钮
        ttk.Button(ctrl, text="Select Image", command=self._select_image
                  ).grid(row=4, column=0, columnspan=2, pady=2, sticky=tk.EW)  # 选择图像
        self.search_btn = ttk.Button(ctrl, text="Search", command=self._start_search, state="disabled")
        self.search_btn.grid(row=5, column=0, columnspan=2, pady=1, sticky=tk.EW)  # 搜索按钮，初始禁用

        ttk.Button(ctrl, text="Histogram", command=self._show_encoding_histogram
                  ).grid(row=6, column=0, columnspan=2, pady=2, sticky=tk.EW)  # 直方图按钮
        

        # 分割线：
        ttk.Separator(ctrl, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=10)
                
        # Reorder1 / Reorder2 采用左右对称布局，共用一行高度
        reorder_row = ttk.Frame(ctrl)
        reorder_row.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=3)
        reorder_row.columnconfigure(0, weight=1)
        reorder_row.columnconfigure(1, weight=1)
        self.reorder_btn1 = ttk.Button(reorder_row, text="Rerank (Liner)", command=self._reorder_results_linear, state="disabled")
        self.reorder_btn1.grid(row=0, column=0, padx=(0, 4), sticky=tk.EW)
        self.reorder_btn2 = ttk.Button(reorder_row, text="Rerank (Graph)", command=self._reorder_results_graph, state="disabled")
        self.reorder_btn2.grid(row=0, column=1, padx=(4, 0), sticky=tk.EW)
        
        qe_row = ttk.Frame(ctrl)
        qe_row.grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=1)
        qe_row.columnconfigure(2, weight=1)
        ttk.Label(qe_row, text="top").grid(row=0, column=0, padx=(0, 2))
        self.qe_k_var = tk.IntVar(value=5)
        ttk.Combobox(qe_row, textvariable=self.qe_k_var, values=list(range(1, 11)),
                     width=3, state="readonly").grid(row=0, column=1, padx=(0, 4))
        self.qe_btn = ttk.Button(qe_row, text="Query Expansion", command=self._query_expansion, state="disabled")
        self.qe_btn.grid(row=0, column=2, sticky=tk.EW)

        # 分割线：
        ttk.Separator(ctrl, orient="horizontal").grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=(10,3))
          
        # 状态显示标签
        self.status_var = tk.StringVar(value="Loading database...")
        ttk.Label(ctrl, textvariable=self.status_var, foreground="white", wraplength=200
                 ).grid(row=11, column=0, columnspan=2, pady=3)
        
        # PR曲线显示区域（右侧）- 先pack以确保显示
        prf = ttk.LabelFrame(top, text="PR Curve", padding=5, width=360)
        prf.pack(side=tk.RIGHT, fill=tk.Y)
        prf.pack_propagate(False)
        # 创建matplotlib图形并嵌入Tkinter
        self.pr_figure = Figure(figsize=(4, 3.5), dpi=80)
        self.pr_ax = self.pr_figure.add_subplot(111)
        self._init_pr_plot()
        self.pr_canvas = FigureCanvasTkAgg(self.pr_figure, prf)
        self.pr_canvas.draw()
        self.pr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 检索信息显示区域（中间，填充剩余空间）
        inf = ttk.LabelFrame(top, text="Search Info", padding=10)
        inf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 多行文本框显示检索结果详情
        self.info_text = tk.Text(inf, wrap=tk.WORD, font=('Arial', 10), height=15)
        scroll = ttk.Scrollbar(inf, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ----- 下部区域：检索结果展示（2行×5列=10张图） -----
        rf = ttk.LabelFrame(main, text="Search Results (Top 10)", padding=5)
        rf.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 创建10个结果槽位，每个槽位包含左侧勾选框+序号车牌号，右侧图像
        self.result_canvases = []
        self.result_vars = []
        self.result_label_vars = []
        self.result_label_widgets = []
        for row in range(2):
            row_frame = ttk.Frame(rf)
            row_frame.pack(fill=tk.X, pady=2, expand=True)
            for _ in range(5):
                item_frame = ttk.Frame(row_frame)
                item_frame.pack(side=tk.LEFT, padx=18, expand=False, fill=tk.BOTH)

                left_panel = ttk.Frame(item_frame, width=50)
                left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2))
                left_panel.pack_propagate(False)

                var = tk.BooleanVar(value=False)
                ttk.Checkbutton(left_panel, variable=var).pack(anchor=tk.W, pady=(6, 1))

                label_var = tk.StringVar(value="")
                lbl = tk.Label(left_panel, textvariable=label_var, justify=tk.LEFT, anchor="w",
                               wraplength=100, font=('Arial', 9, 'bold'))
                lbl.pack(anchor=tk.W, fill=tk.X)

                canvas = tk.Canvas(item_frame, width=185, height=185, bg="lightgray", highlightthickness=0)
                canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

                self.result_canvases.append(canvas)
                self.result_vars.append(var)
                self.result_label_vars.append(label_var)
                self.result_label_widgets.append(lbl)
    
    def _on_feature_toggle(self):
        """Feature单选式切换：SIFT/ORB互斥，默认至少保留一个。"""
        # 始终保持单选：根据当前勾选状态强制互斥
        if self.feature_sift_var.get() and not self.feature_orb_var.get():
            # 选择 SIFT
            self.feature_orb_var.set(False)
            self.algorithm_var.set("SIFT")
        elif self.feature_orb_var.get() and not self.feature_sift_var.get():
            # 选择 ORB
            self.feature_sift_var.set(False)
            self.algorithm_var.set("ORB")
        elif self.feature_sift_var.get() and self.feature_orb_var.get():
            # 若因为点击导致两者都为 True，则根据点击前的 algorithm_var 决定，
            # 默认切换到当前 algorithm_var 的相反项
            if self.algorithm_var.get() == "SIFT":
                self.feature_sift_var.set(False)
                self.algorithm_var.set("ORB")
            else:
                self.feature_orb_var.set(False)
                self.algorithm_var.set("SIFT")
        else:
            # 不允许两个都为 False，回退到上一次选择
            if self.algorithm_var.get() == "ORB":
                self.feature_orb_var.set(True)
            else:
                self.feature_sift_var.set(True)

    def _on_encoding_toggle(self):
        """Encoding单选式切换：BoF/VLAD互斥，默认至少保留一个。"""
        # 始终保持单选：根据当前勾选状态强制互斥
        if self.encoding_bof_var.get() and not self.encoding_vlad_var.get():
            # 选择 BoF
            self.encoding_vlad_var.set(False)
            self.encoding_var.set("BoF")
        elif self.encoding_vlad_var.get() and not self.encoding_bof_var.get():
            # 选择 VLAD
            self.encoding_bof_var.set(False)
            self.encoding_var.set("VLAD")
        elif self.encoding_bof_var.get() and self.encoding_vlad_var.get():
            # 若因为点击导致两者都为 True，则根据点击前的 encoding_var 决定，
            # 默认切换到当前 encoding_var 的相反项
            if self.encoding_var.get() == "BoF":
                self.encoding_bof_var.set(False)
                self.encoding_var.set("VLAD")
            else:
                self.encoding_vlad_var.set(False)
                self.encoding_var.set("BoF")
        else:
            # 不允许两个都为 False，回退到上一次选择
            if self.encoding_var.get() == "VLAD":
                self.encoding_vlad_var.set(True)
            else:
                self.encoding_bof_var.set(True)

    def _get_selected_feature(self):
        """返回当前选中的特征算法字符串("ORB" 或 "SIFT")。"""
        return "ORB" if self.feature_orb_var.get() else "SIFT"

    def _get_selected_encoding(self):
        """返回当前选中的编码方法字符串("VLAD" 或 "BoF")。"""
        return "VLAD" if self.encoding_vlad_var.get() else "BoF"

    def _init_pr_plot(self):
        """初始化空的PR曲线图"""
        self.pr_ax.set_xlabel('Recall')
        self.pr_ax.set_ylabel('Precision')
        self.pr_ax.set_title('PR Curve')
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1)
        self.pr_ax.grid(True, alpha=0.3)
        self.pr_figure.tight_layout()
    
    # =========================================================================
    # 缓存管理
    # =========================================================================
    def _get_database_hash(self):
        """
        计算数据库文件夹的哈希值，用于检测数据库是否发生变化。
        基于相对路径计算，保证项目被复制/移动后哈希不变。详见
        模块级函数 compute_database_hash。
        """
        return compute_database_hash(self.image_folder)

    def _to_rel(self, abs_path):
        """将数据库内的绝对路径转换为相对于image_folder的相对路径（用于写入缓存）"""
        return os.path.relpath(abs_path, self.image_folder)

    def _to_abs(self, rel_path):
        """将缓存中的相对路径还原为当前image_folder下的绝对路径（用于读取缓存）"""
        return os.path.normpath(os.path.join(self.image_folder, rel_path))
    
    def _save_cache(self, all_paths, all_labels, label_counts):
        """
        保存预计算数据到缓存文件。

        为保证项目被复制/移动后缓存仍可用，所有图像路径均以"相对于
        image_folder的相对路径"形式存储，加载时再还原为当前绝对路径。
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # 准备索引缓存数据（只保存encodings/labels/paths，不保存knn对象）
        # paths转为相对路径
        index_data = {}
        for key, retriever in self.index_cache.items():
            index_data[key] = {
                'encodings': retriever.encodings,
                'labels': retriever.labels,
                'paths': [self._to_rel(p) for p in retriever.paths]
            }

        # database_data的键(绝对路径)转为相对路径
        database_data_rel = {self._to_rel(p): v for p, v in self.database_data.items()}

        cache_data = {
            'version': CACHE_FORMAT_VERSION,
            'hash': self._get_database_hash(),
            'database_data': database_data_rel,
            'sift_codebook': self.encoder.codebook,
            'sift_idf': self.encoder.idf_weights,
            'orb_codebook': self.orb_encoder.codebook,
            'orb_idf': self.orb_encoder.idf_weights,
            'index_data': index_data,
            'all_paths': [self._to_rel(p) for p in all_paths],
            'all_labels': all_labels,
            'label_counts': label_counts
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved to {self.cache_file}")
    
    def _load_cache(self):
        """
        从缓存文件加载预计算数据。

        缓存中的路径以相对路径存储，加载时还原为当前image_folder下的绝对路径，
        因此项目被复制/移动到新位置后，缓存依然能直接使用，无需重新预处理。

        为兼容旧版（version<2，存储绝对路径）缓存：若检测到旧格式，则尝试把
        绝对路径按"标签/文件名"重映射到当前目录；只要图像内容未变（哈希基于
        相对路径+大小），即可直接复用，避免重新提取特征。

        返回:
            True: 缓存有效并成功加载
            False: 缓存不存在或已过期
        """
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            version = cache_data.get('version', 1)

            # 路径还原函数：v2+用相对路径直接拼接；v1(旧)用绝对路径取"标签/文件名"重映射
            if version >= CACHE_FORMAT_VERSION:
                to_abs = self._to_abs
            else:
                def to_abs(p):
                    label = os.path.basename(os.path.dirname(p))
                    fn = os.path.basename(p)
                    return os.path.normpath(os.path.join(self.image_folder, label, fn))

            # 检查哈希值，确保数据库没有变化
            # 旧版缓存的hash基于绝对路径，移动后必然不匹配，因此对旧版改用
            # "文件存在性 + 数量一致"作为有效性判断，从而仍能复用其特征数据。
            if version >= CACHE_FORMAT_VERSION:
                if cache_data.get('hash') != self._get_database_hash():
                    print("Database changed, cache invalidated")
                    return False
            else:
                cached_rel = sorted(
                    os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                    for p in cache_data.get('all_paths', [])
                )
                current_rel = []
                for root, dirs, files in os.walk(self.image_folder):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for fn in files:
                        if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            current_rel.append(os.path.join(
                                os.path.basename(root), fn))
                current_rel.sort()
                if cached_rel != current_rel:
                    print("Database changed, legacy cache invalidated")
                    return False

            # 恢复数据（database_data的键还原为绝对路径）
            self.database_data = {to_abs(p): v for p, v in cache_data['database_data'].items()}
            self.encoder.codebook = cache_data['sift_codebook']
            self.encoder.idf_weights = cache_data['sift_idf']
            self.orb_encoder.codebook = cache_data['orb_codebook']
            self.orb_encoder.idf_weights = cache_data['orb_idf']

            # 重建检索索引（需要重新fit KNN模型），paths还原为绝对路径
            for key, data in cache_data['index_data'].items():
                retriever = ImageRetriever(metric="cosine")
                retriever.encodings = data['encodings']
                retriever.labels = data['labels']
                retriever.paths = [to_abs(p) for p in data['paths']]
                from sklearn.neighbors import NearestNeighbors
                retriever.knn = NearestNeighbors(metric=retriever.metric, algorithm='brute')
                retriever.knn.fit(retriever.encodings)
                self.index_cache[key] = retriever

            self.all_paths = [to_abs(p) for p in cache_data['all_paths']]
            self.all_labels = cache_data['all_labels']
            self.evaluator = PerformanceEvaluator(cache_data['label_counts'])

            print(f"Cache loaded from {self.cache_file}")

            # 旧版缓存：加载成功后立即以新格式(相对路径)重写，后续移动即可零成本复用
            if version < CACHE_FORMAT_VERSION:
                try:
                    self._save_cache(self.all_paths, self.all_labels,
                                     cache_data['label_counts'])
                    print("Legacy cache upgraded to portable format (v%d)" % CACHE_FORMAT_VERSION)
                except Exception as e:
                    print(f"Cache upgrade skipped: {e}")

            return True

        except Exception as e:
            print(f"Cache load failed: {e}")
            return False
    
    # =========================================================================
    # 数据库加载（异步执行，并行构建索引）
    # =========================================================================
    def _load_database_async(self):
        """
        异步加载数据库并预计算所有编码组合
        
        优先从缓存加载，若缓存无效则重新计算并保存缓存
        加载过程在后台线程执行，不阻塞GUI：
        1. 尝试加载缓存
        2. 若缓存无效：提取所有图像的SIFT和ORB特征
        3. 使用K-means构建视觉词典
        4. 并行预计算8种编码组合的检索索引
        5. 保存缓存供下次使用
        """
        def load():
            try:
                # 首先尝试从缓存加载
                self.root.after(0, lambda: self.status_var.set("Checking cache..."))
                if self._load_cache():
                    self.database_loaded = True
                    self.root.after(0, self._on_database_loaded)
                    return
                
                # 缓存无效，重新计算
                self.root.after(0, lambda: self.status_var.set("Cache miss, extracting features..."))
                
                # 加载图像数据库（按文件夹结构）
                db = load_image_database(self.image_folder)
                total = sum(len(imgs) for imgs in db.values())
                count = 0
                all_sift_desc, all_labels, all_paths = [], [], []
                
                # 步骤1：提取所有图像的特征
                for label, paths in db.items():
                    for path in paths:
                        img = cv2.imread(path)
                        if img is None:
                            continue
                        # 同时提取SIFT和ORB特征，便于切换算法
                        _, sift_desc, _ = self.extractor.extract(img, "SIFT")
                        _, orb_desc, _ = self.extractor.extract(img, "ORB")
                        
                        # 缓存特征数据
                        self.database_data[path] = {'label': label, 'sift_desc': sift_desc, 'orb_desc': orb_desc}
                        if sift_desc is not None:
                            all_sift_desc.append(sift_desc)
                            all_labels.append(label)
                            all_paths.append(path)
                        count += 1
                        # 每处理20张图更新一次状态
                        if count % 20 == 0:
                            self.root.after(0, lambda c=count, t=total: self.status_var.set(f"Extracting: {c}/{t}"))
                
                # 步骤2：分别为SIFT和ORB构建视觉词典
                # SIFT和ORB描述子维度不同(128 vs 32)，需要各自独立的codebook
                self.root.after(0, lambda: self.status_var.set("Building SIFT codebook..."))
                self.encoder.build_codebook(all_sift_desc)
                self.encoder.compute_idf(all_sift_desc)
                
                # 为ORB构建单独的codebook
                self.root.after(0, lambda: self.status_var.set("Building ORB codebook..."))
                all_orb_desc = [self.database_data[p]['orb_desc'] for p in all_paths 
                               if self.database_data[p]['orb_desc'] is not None]
                if all_orb_desc:
                    self.orb_encoder.build_codebook(all_orb_desc)
                    self.orb_encoder.compute_idf(all_orb_desc)
                
                # 步骤3：并行预计算所有编码组合的检索索引
                # 8种组合：SIFT/ORB × BoF/VLAD × 有无IDF
                combos = [
                    ("SIFT", "BoF", False), ("SIFT", "BoF", True),
                    ("SIFT", "VLAD", False), ("SIFT", "VLAD", True),
                    ("ORB", "BoF", False), ("ORB", "BoF", True),
                    ("ORB", "VLAD", False), ("ORB", "VLAD", True)
                ]
                
                def build_index(combo):
                    """为单个编码组合构建检索索引"""
                    algo, method, use_idf = combo
                    # 根据算法选择对应的编码器和描述子
                    if algo == "SIFT":
                        encoder = self.encoder
                        desc_key = 'sift_desc'
                    else:
                        encoder = self.orb_encoder
                        desc_key = 'orb_desc'
                    # 编码所有数据库图像
                    encodings = [encoder.encode(self.database_data[p][desc_key], method, use_idf)
                                for p in all_paths]
                    retriever = ImageRetriever(metric="cosine")
                    retriever.build_index(encodings, all_labels, all_paths)
                    return combo, retriever
                
                self.root.after(0, lambda: self.status_var.set("Building indexes..."))
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for combo, retriever in executor.map(build_index, combos):
                        self.index_cache[combo] = retriever
                
                # 初始化性能评估器（需要每个类别的图像数量来计算召回率）
                label_counts = {lbl: all_labels.count(lbl) for lbl in set(all_labels)}
                self.evaluator = PerformanceEvaluator(label_counts)
                self.all_paths, self.all_labels = all_paths, all_labels
                
                # 保存缓存供下次使用
                self.root.after(0, lambda: self.status_var.set("Saving cache..."))
                self._save_cache(all_paths, all_labels, label_counts)
                
                # 标记加载完成，触发回调
                self.database_loaded = True
                self.root.after(0, self._on_database_loaded)
                
            except Exception as e:
                import traceback; traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Load failed: {e}"))
        
        # 启动后台线程执行加载（daemon=True表示主程序退出时自动终止）
        threading.Thread(target=load, daemon=True).start()
    
    def _on_database_loaded(self):
        """数据库加载完成的回调函数，启用搜索按钮"""
        self.status_var.set(f"Database loaded: {len(self.database_data)} images")
        self.search_btn.config(state="normal")  # 启用搜索按钮
        self.qe_btn.config(state="normal")
        self.reorder_btn1.config(state="normal")
        self.reorder_btn2.config(state="normal")
    
    # =========================================================================
    # 图像选择与显示
    # =========================================================================
    def _select_image(self):
        """
        打开文件对话框选择查询图像
        
        选择后会在查询图像区域显示，并清空之前的检索结果
        """
        path = filedialog.askopenfilename(
            title="Select Query Image", initialdir=self.test_folder,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.query_image = cv2.imread(path)  # 读取图像
            self.query_path = path
            self._display_image(self.query_image, self.query_canvas)  # 显示在查询区域
            self.status_var.set(f"{os.path.basename(path)}")
            # 清空结果显示区域
            for canvas in self.result_canvases:
                canvas.delete("all")
            for var in self.result_vars:
                var.set(False)
            for label_var, label_widget in zip(self.result_label_vars, self.result_label_widgets):
                label_var.set("")
                label_widget.config(foreground="black")
            self.info_text.delete(1.0, tk.END)
    
    def _display_image(self, cv_img, canvas, label=None):
        """
        在Canvas控件上显示OpenCV图像
        
        参数:
            cv_img: OpenCV图像（BGR格式）
            canvas: 目标Canvas控件
            label: 可选的标签文字，显示在图像下方
        """
        if cv_img is None:
            return
        
        # 将BGR转换为RGB，OpenCV默认是BGR，PIL需要RGB
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # 获取Canvas尺寸，按比例缩放图像以适应
        canvas.update()
        w, h = canvas.winfo_width() or 220, canvas.winfo_height() or 200
        pil_img.thumbnail((w, h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # 在Canvas左上角绘制图像，尽量贴边
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img  # 保持引用，防止被垃圾回收
        
        # 如果有标签，在底部绘制文字
        if label:
            canvas.create_text(max(10, pil_img.width // 2), max(10, h - 8), text=label, font=('Arial', 8))
    
    # =========================================================================
    # 图像检索
    # =========================================================================
    def _get_query_encoding(self, algo, method, use_idf, image=None):
        """
        提取并编码查询图像，得到一个全局向量表示。

        参数:
            algo: 特征算法("SIFT"/"ORB")
            method: 编码方法("BoF"/"VLAD")
            use_idf: 是否使用 TF-IDF 加权
            image: 要编码的图像；不传则用当前查询图像 self.query_image
        返回:
            (编码向量, 特征提取耗时秒数)
        """
        # 没传图就用当前查询图
        image = image if image is not None else self.query_image
        # 提取局部特征描述子(同时拿到耗时，用于界面显示)
        _, desc, ext_time = self.extractor.extract(image, algo)
        # SIFT 和 ORB 各有独立的编码器(码本不同)，按算法选对应的那个
        encoder = self.encoder if algo == "SIFT" else self.orb_encoder
        return encoder.encode(desc, method, use_idf), ext_time

    def _search_with_encoding(self, query_enc, algo, method, use_idf, exclude_path=None, k=10):
        """
        基于给定查询向量执行检索(直接复用预计算好的索引，速度极快)。

        参数:
            query_enc: 查询向量
            algo/method/use_idf: 用来定位预建好的索引(8 种组合之一)
            exclude_path: 要排除的路径(通常是查询图自己)
            k: 返回结果数量
        返回:
            (结果列表, 检索耗时秒数)
        """
        # 用(算法,编码,IDF)三元组作为键，取出对应的预建索引直接搜
        key = (algo, method, use_idf)
        return self.index_cache[key].search(query_enc, k=k, exclude_path=exclude_path)

    def _start_search(self):
        """
        执行图像检索
        
        使用预计算的索引，实现毫秒级快速响应
        """
        if not self.database_loaded or self.query_image is None:
            messagebox.showwarning("Warning", "Please load database and select image first")
            return
        
        self.status_var.set("Searching...")
        self.info_text.delete(1.0, tk.END)
        
        # 获取用户选择的算法和编码方法
        algo, method, use_idf = self._get_selected_feature(), self._get_selected_encoding(), self.use_idf_var.get()
        
        # 提取查询图像的特征并编码（根据算法选择对应的编码器）
        query_enc, ext_time = self._get_query_encoding(algo, method, use_idf)
        
        # 使用预计算的索引进行检索
        results, search_time = self._search_with_encoding(query_enc, algo, method, use_idf, exclude_path=self.query_path, k=10)
        
        self.current_results = results
        self.query_expanded = False
        self.reorder_btn1.config(state="normal")
        self.reorder_btn2.config(state="normal")
        self._show_results(results, ext_time, search_time, algo, method, use_idf)

    def _show_results(self, results, ext_time, search_time, algo, method, use_idf):
        """
        显示检索结果
        
        在10个Canvas上显示Top10结果图像：
        - 绿色边框：与查询图像同类别（正确匹配）
        - 红色边框：不同类别（错误匹配）
        """
        query_label = os.path.basename(os.path.dirname(self.query_path))

        # 清空所有结果槽位
        for i in range(len(self.result_canvases)):
            self.result_vars[i].set(False)
            self.result_label_vars[i].set("")
            self.result_label_widgets[i].config(foreground="black")
            self.result_canvases[i].delete("all")

        # 显示Top10结果图像，并用彩色边框标注正确/错误
        for i, canvas in enumerate(self.result_canvases):
            if i < len(results):
                r = results[i]
                color = "green" if r['label'] == query_label else "red"
                self.result_label_vars[i].set(f"{i+1}. {r['label']}")
                self.result_label_widgets[i].config(foreground=color)
                self._display_image(cv2.imread(r['path']), canvas)
                canvas.update()
                canvas.create_rectangle(1, 1, canvas.winfo_width()-1, canvas.winfo_height()-1,
                                        outline=color, width=3)
            else:
                canvas.delete("all")

        retrieved = [r['label'] for r in results]
        precision = sum(1 for l in retrieved if l == query_label) / len(retrieved) if retrieved else 0

        idf_str = " (TF-IDF)" if use_idf else ""
        info = f"=== Search Results ===\n"
        info += f"Method: {algo} + {method}{idf_str}\n"
        info += f"Query: {os.path.basename(self.query_path)}\n"
        info += f"True Label: {query_label}\n\n"
        info += f"Feature: {ext_time*1000:.1f}ms | Search: {search_time*1000:.1f}ms\n"
        info += f"Top10 Precision: {precision*100:.1f}%\n\n"
        info += "=== Top10 Results ===\n"
        for i, r in enumerate(results):
            mark = "Y" if r['label'] == query_label else "N"
            info += f"{i+1}. [{mark}] {r['label']} (dist:{r['distance']:.4f})\n"

        self.info_text.insert(tk.END, info)
        self.status_var.set(f"Done. Precision: {precision*100:.1f}%")
        self._plot_single_query_pr(results, query_label)
    
    # =========================================================================
    # 测试集评估
    # =========================================================================
    def _run_evaluation(self):
        """
        在测试集上运行完整评估
        
        对测试集中的每张图像执行检索，计算各项性能指标，在后台线程执行
        """
        if not self.database_loaded:
            messagebox.showwarning("Warning", "Please wait for database to load")
            return
        
        algo, method, use_idf = self.algorithm_var.get(), self.encoding_var.get(), self.use_idf_var.get()
        self.status_var.set(f"Evaluating ({algo}+{method})...")
        
        def evaluate():
            """后台评估线程的执行函数"""
            try:
                # 获取对应的检索器和编码器
                key = (algo, method, use_idf)
                retriever = self.index_cache[key]
                encoder = self.encoder if algo == "SIFT" else self.orb_encoder
                test_db = load_image_database(self.test_folder)
                
                results_list, times = [], []
                total = sum(len(p) for p in test_db.values())
                count = 0
                
                # 对测试集中的每张图像执行检索
                for label, paths in test_db.items():
                    for path in paths:
                        img = cv2.imread(path)
                        if img is None:
                            continue
                        # 提取特征并编码
                        _, desc, _ = self.extractor.extract(img, algo)
                        if desc is None:
                            continue
                        enc = encoder.encode(desc, method, use_idf)
                        # 执行检索
                        res, t = retriever.search(enc, k=10, exclude_path=path)
                        results_list.append((res, label))
                        times.append(t)
                        count += 1
                        # 每处理10张图更新状态
                        if count % 10 == 0:
                            self.root.after(0, lambda c=count, t=total: self.status_var.set(f"Evaluating: {c}/{t}"))
                
                # 计算评估指标
                metrics = self.evaluator.evaluate(results_list)
                self.root.after(0, lambda: self._show_evaluation(metrics, np.mean(times)*1000, algo, method, use_idf))
                
            except Exception as e:
                import traceback; traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Evaluation failed: {e}"))
        
        # 启动后台评估线程
        threading.Thread(target=evaluate, daemon=True).start()
    
    def _show_evaluation(self, metrics, avg_time, algo, method, use_idf):
        """
        显示评估结果
        
        在信息文本框中显示Precision、Recall、mAP等各项指标
        """
        idf_str = " (TF-IDF)" if use_idf else ""
        info = f"=== Evaluation Results ===\nMethod: {algo} + {method}{idf_str}\n\n"
        
        # 显示Top1、Top5、Top10的关键指标
        for k in [1, 5, 10]:
            info += f"--- Top{k} ---\n"
            info += f"  P: {metrics[f'Precision@{k}']*100:.1f}%  R: {metrics[f'Recall@{k}']*100:.1f}%  mAP: {metrics[f'mAP@{k}']*100:.1f}%\n"
        
        # 显示mAP@1到mAP@10的完整统计
        info += "\n=== mAP@1~10 ===\n"
        maps = [metrics.get(f'mAP@{k}', 0) * 100 for k in range(1, 11)]
        for k, m in enumerate(maps, 1):
            info += f"  mAP@{k}: {m:.1f}%\n"
        info += f"\nAvg mAP: {sum(maps)/10:.1f}%\nAvg Time: {avg_time:.1f}ms\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.status_var.set("Evaluation Done")
        # 绘制测试集的PR曲线
        self._plot_pr_curve(metrics['pr_curve'])
    
    # =========================================================================
    # PR曲线绘制
    # =========================================================================
    def _plot_pr_curve(self, pr_data):
        """
        绘制完整测试集评估的PR曲线
        
        横轴为召回率(Recall)，纵轴为精度(Precision)
        曲线越靠近右上角说明性能越好
        """
        self.pr_ax.clear()
        self.pr_ax.plot(pr_data['recall'], pr_data['precision'], 'b-', linewidth=2)
        self.pr_ax.set_xlabel('Recall')
        self.pr_ax.set_ylabel('Precision')
        self.pr_ax.set_title('PR Curve (Test Set)')
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1)
        self.pr_ax.grid(True, alpha=0.3)
        self.pr_figure.tight_layout()
        self.pr_canvas.draw()
    
    def _plot_single_query_pr(self, results, query_label):
        """
        绘制单个查询的PR曲线
        
        展示当前查询在不同K值下的精度和召回率变化
        每个点对应一个K值（1到10）
        """
        labels = [r['label'] for r in results]
        # 获取该类别在数据库中的总数（用于计算召回率的分母）
        total_rel = self.evaluator.label_counts.get(query_label, 1)
        
        precs, recs = [], []
        for k in range(1, len(labels) + 1):
            # 计算前k个结果中的正确数
            rel = sum(1 for l in labels[:k] if l == query_label)
            precs.append(rel / k)  # Precision@k
            recs.append(rel / total_rel if total_rel > 0 else 0)  # Recall@k
        
        self.pr_ax.clear()
        # 用红色圆点标记每个K值的位置
        self.pr_ax.plot(recs, precs, 'r-o', linewidth=2, markersize=4)
        self.pr_ax.set_xlabel('Recall')
        self.pr_ax.set_ylabel('Precision')
        self.pr_ax.set_title(f'PR Curve ({query_label})')
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1)
        self.pr_ax.grid(True, alpha=0.3)
        self.pr_figure.tight_layout()
        self.pr_canvas.draw()
    
    # =========================================================================
    # 特征编码直方图可视化
    # =========================================================================
    def _show_encoding_histogram(self):
        """
        显示特征编码直方图
        
        在新窗口中显示查询图像和Top10检索结果的特征编码直方图：
        - 第一行：查询图像的编码
        - 第二行：Top1-5的编码对比
        - 第三行：Top6-10的编码对比
        
        相似的图像应该有相似的直方图分布
        """
        # 检查前置条件
        if not self.database_loaded:
            return messagebox.showwarning("Warning", "Please wait for database to load")
        if self.query_image is None:
            return messagebox.showwarning("Warning", "Please select an image first")
        if self.current_results is None:
            return messagebox.showwarning("Warning", "Please run search first")
        
        method, use_idf = self.encoding_var.get(), self.use_idf_var.get()
        algo = self.algorithm_var.get()
        
        # 根据算法选择对应的编码器
        encoder = self.encoder if algo == "SIFT" else self.orb_encoder
        
        # 获取查询图像的编码
        _, desc, _ = self.extractor.extract(self.query_image, algo)
        query_enc = encoder.encode(desc, method, use_idf)
        
        # 获取Top10结果的编码
        desc_key = 'sift_desc' if algo == "SIFT" else 'orb_desc'
        res_encs, res_labels = [], []
        for r in self.current_results[:10]:
            d = self.database_data.get(r['path'], {}).get(desc_key)
            if d is not None:
                res_encs.append(encoder.encode(d, method, use_idf))
                res_labels.append(r['label'])
        
        # 创建新窗口显示直方图
        win = tk.Toplevel(self.root)
        win.title("Feature Encoding Histogram")
        win.geometry("1200x800")
        
        fig = Figure(figsize=(14, 9), dpi=90)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # 5种不同颜色
        x = np.arange(len(query_enc))  # 视觉词索引
        
        # 第一行：查询图像的编码直方图
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.bar(x, query_enc, color='steelblue', alpha=0.8)
        ax1.set_title(f'Query Encoding ({method}{" + TF-IDF" if use_idf else ""})')
        ax1.set_xlabel('Visual Word Index')
        
        # 第二行：Top1-5结果的编码对比
        ax2 = fig.add_subplot(3, 1, 2)
        for i, (enc, lbl) in enumerate(zip(res_encs[:5], res_labels[:5])):
            ax2.bar(x + i*0.15, enc, 0.15, label=f'{i+1}.{lbl}', color=colors[i], alpha=0.7)
        ax2.set_title('Top 1-5 Encoding')
        ax2.legend(loc='upper right', fontsize=8)
        
        # 第三行：Top6-10结果的编码对比
        ax3 = fig.add_subplot(3, 1, 3)
        for i, (enc, lbl) in enumerate(zip(res_encs[5:], res_labels[5:])):
            ax3.bar(x + i*0.15, enc, 0.15, label=f'{i+6}.{lbl}', color=colors[i], alpha=0.7)
        ax3.set_title('Top 6-10 Encoding')
        ax3.legend(loc='upper right', fontsize=8)
        
        fig.tight_layout()
        FigureCanvasTkAgg(fig, win).get_tk_widget().pack(fill=tk.BOTH, expand=True)
    

def main():
    """
    启动GUI应用程序
    
    从data文件夹中读取image和test数据集
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # 创建主窗口并启动应用
    root = tk.Tk()
    ImageSearchApp(root, os.path.join(data_dir, "image"), os.path.join(data_dir, "test"))
    root.mainloop()  # 进入Tkinter事件循环


if __name__ == "__main__":
    main()
