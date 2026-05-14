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


class ImageSearchApp:
    """
    图像检索系统GUI应用主类
    
    该类整合了所有GUI组件和检索功能，实现完整的交互式图像检索系统。
    启动时会异步加载数据库并预计算所有编码组合，以实现快速响应。
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
        self.database_data = {}        # 数据库数据缓存 {路径: {label, sift_desc, orb_desc}}
        self.index_cache = {}          # 预计算的检索索引缓存 {(算法, 编码, IDF): retriever}
        
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
        qf = ttk.LabelFrame(top, text="Query Image", padding=5, width=380)
        qf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        qf.pack_propagate(False)
        self.query_canvas = tk.Canvas(qf, bg="white")  # 用Canvas显示图像
        self.query_canvas.pack(fill=tk.BOTH, expand=True)

        # 控制面板（中左）- 包含算法选择、按钮等
        ctrl = ttk.LabelFrame(top, text="Control Panel", padding=20, width=300)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ctrl.pack_propagate(False)
        
        # 特征算法下拉框：SIFT或ORB
        ttk.Label(ctrl, text="Feature:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(ctrl, textvariable=self.algorithm_var, values=["SIFT", "ORB"], 
                    state="readonly", width=12).grid(row=0, column=1, pady=5)
        # 编码方法下拉框：BoF或VLAD
        ttk.Label(ctrl, text="Encoding:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(ctrl, textvariable=self.encoding_var, values=["BoF", "VLAD"], 
                    state="readonly", width=12).grid(row=1, column=1, pady=5)
        # TF-IDF开关复选框
        ttk.Checkbutton(ctrl, text="Enable TF-IDF", variable=self.use_idf_var
                       ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 功能按钮
        ttk.Button(ctrl, text="Select Image", command=self._select_image
                  ).grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)  # 选择图像
        self.search_btn = ttk.Button(ctrl, text="Search", command=self._start_search, state="disabled")
        self.search_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)  # 搜索按钮，初始禁用
        self.reorder_btn = ttk.Button(ctrl, text="Reorder", command=self._reorder_results, state="disabled")
        self.reorder_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky=tk.EW)  # 重排序按钮
        ttk.Button(ctrl, text="Evaluate", command=self._run_evaluation
                  ).grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.EW)  # 评估按钮
        ttk.Button(ctrl, text="Histogram", command=self._show_encoding_histogram
                  ).grid(row=7, column=0, columnspan=2, pady=5, sticky=tk.EW)  # 直方图按钮
        
        # 状态显示标签
        self.status_var = tk.StringVar(value="Loading database...")
        ttk.Label(ctrl, textvariable=self.status_var, foreground="blue", wraplength=200
                 ).grid(row=8, column=0, columnspan=2, pady=10)
        
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
        
        # 创建10个Canvas用于显示检索结果图像
        self.result_canvases = []
        for row in range(2):
            row_frame = ttk.Frame(rf)
            row_frame.pack(fill=tk.X, pady=2, expand=True)
            for _ in range(5):
                canvas = tk.Canvas(row_frame, width=220, height=200, bg="lightgray")
                canvas.pack(side=tk.LEFT, padx=5, expand=True)
                self.result_canvases.append(canvas)
    
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
        计算数据库文件夹的哈希值，用于检测数据库是否发生变化
        基于所有图像文件的路径和修改时间生成哈希
        """
        hash_data = []
        for root, dirs, files in os.walk(self.image_folder):
            dirs[:] = [d for d in dirs if not d.startswith('.')]  # 跳过隐藏文件夹
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    filepath = os.path.join(root, f)
                    mtime = os.path.getmtime(filepath)
                    hash_data.append(f"{filepath}:{mtime}")
        return hashlib.md5('\n'.join(hash_data).encode()).hexdigest()
    
    def _save_cache(self, all_paths, all_labels, label_counts):
        """
        保存预计算数据到缓存文件
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 准备索引缓存数据（只保存encodings/labels/paths，不保存knn对象）
        index_data = {}
        for key, retriever in self.index_cache.items():
            index_data[key] = {
                'encodings': retriever.encodings,
                'labels': retriever.labels,
                'paths': retriever.paths
            }
        
        cache_data = {
            'hash': self._get_database_hash(),
            'database_data': self.database_data,
            'sift_codebook': self.encoder.codebook,
            'sift_idf': self.encoder.idf_weights,
            'orb_codebook': self.orb_encoder.codebook,
            'orb_idf': self.orb_encoder.idf_weights,
            'index_data': index_data,
            'all_paths': all_paths,
            'all_labels': all_labels,
            'label_counts': label_counts
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved to {self.cache_file}")
    
    def _load_cache(self):
        """
        从缓存文件加载预计算数据
        
        返回:
            True: 缓存有效并成功加载
            False: 缓存不存在或已过期
        """
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查哈希值，确保数据库没有变化
            if cache_data.get('hash') != self._get_database_hash():
                print("Database changed, cache invalidated")
                return False
            
            # 恢复数据
            self.database_data = cache_data['database_data']
            self.encoder.codebook = cache_data['sift_codebook']
            self.encoder.idf_weights = cache_data['sift_idf']
            self.orb_encoder.codebook = cache_data['orb_codebook']
            self.orb_encoder.idf_weights = cache_data['orb_idf']
            
            # 重建检索索引（需要重新fit KNN模型）
            for key, data in cache_data['index_data'].items():
                retriever = ImageRetriever(metric="cosine")
                retriever.encodings = data['encodings']
                retriever.labels = data['labels']
                retriever.paths = data['paths']
                from sklearn.neighbors import NearestNeighbors
                retriever.knn = NearestNeighbors(metric=retriever.metric, algorithm='brute')
                retriever.knn.fit(retriever.encodings)
                self.index_cache[key] = retriever
            
            self.all_paths = cache_data['all_paths']
            self.all_labels = cache_data['all_labels']
            self.evaluator = PerformanceEvaluator(cache_data['label_counts'])
            
            print(f"Cache loaded from {self.cache_file}")
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
            self.status_var.set(f"Selected: {os.path.basename(path)}")
            # 清空结果显示区域
            for canvas in self.result_canvases:
                canvas.delete("all")
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
        pil_img.thumbnail((w - 4, h - 20 if label else h - 4), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # 在Canvas上居中绘制图像
        canvas.delete("all")
        canvas.create_image((w - pil_img.width) // 2, 2, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img  # 保持引用，防止被垃圾回收
        
        # 如果有标签，在底部绘制文字
        if label:
            canvas.create_text(w // 2, h - 10, text=label, font=('Arial', 8))
    
    # =========================================================================
    # 图像检索
    # =========================================================================
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
        algo, method, use_idf = self.algorithm_var.get(), self.encoding_var.get(), self.use_idf_var.get()
        
        # 提取查询图像的特征并编码（根据算法选择对应的编码器）
        _, desc, ext_time = self.extractor.extract(self.query_image, algo)
        encoder = self.encoder if algo == "SIFT" else self.orb_encoder
        query_enc = encoder.encode(desc, method, use_idf)
        
        # 使用预计算的索引进行检索
        key = (algo, method, use_idf)
        results, search_time = self.index_cache[key].search(query_enc, k=10, exclude_path=self.query_path)
        
        self.current_results = results
        self._show_results(results, ext_time, search_time, algo, method, use_idf)
    
    def _show_results(self, results, ext_time, search_time, algo, method, use_idf):
        """
        显示检索结果
        
        在10个Canvas上显示Top10结果图像：
        - 绿色边框：与查询图像同类别（正确匹配）
        - 红色边框：不同类别（错误匹配）
        """
        # 从路径中提取查询图像的真实标签（父文件夹名）
        query_label = os.path.basename(os.path.dirname(self.query_path))
        
        # 显示Top10结果图像，并用彩色边框标注正确/错误
        for i, canvas in enumerate(self.result_canvases):
            if i < len(results):
                r = results[i]
                self._display_image(cv2.imread(r['path']), canvas, f"{i+1}. {r['label']}")
                canvas.update()
                # 同标签=绿色边框，不同=红色边框
                color = "green" if r['label'] == query_label else "red"
                cw, ch = canvas.winfo_width(), canvas.winfo_height()
                canvas.create_rectangle(2, 2, cw-2, ch-2, outline=color, width=3)
            else:
                canvas.delete("all")
        
        # 计算Top10精度
        retrieved = [r['label'] for r in results]
        precision = sum(1 for l in retrieved if l == query_label) / len(retrieved)
        
        # 在信息文本框显示检索详情
        idf_str = " (TF-IDF)" if use_idf else ""
        info = f"=== Search Results ===\n"
        info += f"Method: {algo} + {method}{idf_str}\n"
        info += f"Query: {os.path.basename(self.query_path)}\n"
        info += f"True Label: {query_label}\n\n"
        info += f"Feature: {ext_time*1000:.1f}ms | Search: {search_time*1000:.1f}ms\n"
        info += f"Top10 Precision: {precision*100:.1f}%\n\n"
        info += "=== Top10 Results ===\n"
        for i, r in enumerate(results):
            mark = "Y" if r['label'] == query_label else "N"  # Y=正确，N=错误
            info += f"{i+1}. [{mark}] {r['label']} (dist:{r['distance']:.4f})\n"
        
        self.info_text.insert(tk.END, info)
        self.status_var.set(f"Done. Precision: {precision*100:.1f}%")
        # 启用重排序按钮
        self.reorder_btn.config(state="normal")
        # 绘制该查询的PR曲线
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
    
    # =========================================================================
    # 基于线性组合的图像重排序
    # =========================================================================
    def _extract_color_hist(self, image):
        """
        提取颜色直方图特征
        
        使用HSV颜色空间，分别计算H、S、V通道的直方图并拼接
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # H通道：0-180，S和V通道：0-256
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        # L2归一化
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist
    
    def _extract_texture_lbp(self, image):
        """
        提取LBP纹理特征
        
        使用简化的LBP算法，计算局部二值模式直方图
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # 简化LBP：使用Sobel算子提取纹理
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 计算梯度方向直方图
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi + 180
        hist, _ = np.histogram(angle.flatten(), bins=36, range=(0, 360), weights=magnitude.flatten())
        # L2归一化
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist
    
    def _extract_shape_feature(self, image):
        """
        提取形状特征
        
        使用Hu矩和边缘直方图
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Hu矩（7个不变矩）
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        # 对Hu矩取对数（使数值更稳定）
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 边缘特征
        edges = cv2.Canny(gray, 100, 200)
        # 计算边缘方向直方图
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi + 180
        edge_hist, _ = np.histogram(angle.flatten(), bins=18, range=(0, 360))
        
        # 拼接特征
        feature = np.concatenate([hu_moments, edge_hist.astype(np.float64)])
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature
    
    def _compute_similarity(self, feat1, feat2):
        """计算两个特征向量的余弦相似度"""
        norm1, norm2 = np.linalg.norm(feat1), np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(feat1, feat2) / (norm1 * norm2)
    
    def _reorder_results(self):
        """
        基于线性组合的图像重排序
        
        步骤：
        1. 提取查询图像和检索结果的颜色、纹理、形状特征
        2. 计算各特征的相似度
        3. 将原始检索得分与各特征相似度进行线性组合
        4. 根据组合得分重新排序
        """
        if self.current_results is None or len(self.current_results) == 0:
            messagebox.showwarning("Warning", "Please run search first")
            return
        
        if self.query_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        self.status_var.set("Reordering...")
        
        # 提取查询图像的特征
        query_color = self._extract_color_hist(self.query_image)
        query_texture = self._extract_texture_lbp(self.query_image)
        query_shape = self._extract_shape_feature(self.query_image)
        
        # 线性组合权重：原始得分 + 颜色 + 纹理 + 形状
        w_original, w_color, w_texture, w_shape = 0.7, 0.15, 0.08, 0.07
        
        # 计算原始得分的归一化范围
        if self.current_results:
            max_dist = max(r['distance'] for r in self.current_results)
            min_dist = min(r['distance'] for r in self.current_results)
            dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        
        # 计算每个结果的组合相似度
        reordered = []
        for r in self.current_results:
            img = cv2.imread(r['path'])
            if img is None:
                continue
            
            # 原始检索得分（距离转相似度，归一化）
            orig_sim = 1 - (r['distance'] - min_dist) / dist_range if dist_range > 0 else 1.0
            
            # 提取结果图像的特征
            res_color = self._extract_color_hist(img)
            res_texture = self._extract_texture_lbp(img)
            res_shape = self._extract_shape_feature(img)
            
            # 计算各特征相似度
            sim_color = self._compute_similarity(query_color, res_color)
            sim_texture = self._compute_similarity(query_texture, res_texture)
            sim_shape = self._compute_similarity(query_shape, res_shape)
            
            # 线性组合所有特征得分
            combined_score = (w_original * orig_sim + 
                            w_color * sim_color + 
                            w_texture * sim_texture + 
                            w_shape * sim_shape)
            
            reordered.append({
                'path': r['path'],
                'label': r['label'],
                'distance': 1 - combined_score,  # 转换为距离（越小越好）
                'combined_score': combined_score,
                'orig_sim': orig_sim,
                'sim_color': sim_color,
                'sim_texture': sim_texture,
                'sim_shape': sim_shape
            })
        
        # 按组合得分降序排列（得分越高越相似）
        reordered.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 更新当前结果
        self.current_results = reordered
        
        # 显示重排序结果
        self._show_reordered_results(reordered)
    
    def _show_reordered_results(self, results):
        """
        显示重排序后的结果
        """
        query_label = os.path.basename(os.path.dirname(self.query_path))
        
        # 显示Top10结果图像
        for i, canvas in enumerate(self.result_canvases):
            if i < len(results):
                r = results[i]
                self._display_image(cv2.imread(r['path']), canvas, f"{i+1}. {r['label']}")
                canvas.update()
                color = "green" if r['label'] == query_label else "red"
                cw, ch = canvas.winfo_width(), canvas.winfo_height()
                canvas.create_rectangle(2, 2, cw-2, ch-2, outline=color, width=3)
            else:
                canvas.delete("all")
        
        # 计算Top10精度
        retrieved = [r['label'] for r in results]
        precision = sum(1 for l in retrieved if l == query_label) / len(retrieved) if retrieved else 0
        
        # 更新信息文本
        algo, method, use_idf = self.algorithm_var.get(), self.encoding_var.get(), self.use_idf_var.get()
        idf_str = " (TF-IDF)" if use_idf else ""
        info = f"=== Reordered Results ===\n"
        info += f"Method: {algo} + {method}{idf_str} + Rerank\n"
        info += f"Rerank: Orig(0.5) + Color(0.2) + Tex(0.15) + Shape(0.15)\n"
        info += f"Query: {os.path.basename(self.query_path)}\n"
        info += f"True Label: {query_label}\n\n"
        info += f"Top10 Precision: {precision*100:.1f}%\n\n"
        info += "=== Top10 Results ===\n"
        for i, r in enumerate(results[:10]):
            mark = "Y" if r['label'] == query_label else "N"
            info += f"{i+1}. [{mark}] {r['label']} (score:{r['combined_score']:.4f})\n"
            info += f"    Color:{r['sim_color']:.3f} Tex:{r['sim_texture']:.3f} Shape:{r['sim_shape']:.3f}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.status_var.set(f"Reordered. Precision: {precision*100:.1f}%")
        
        # 绘制重排序后的PR曲线
        self._plot_single_query_pr(results, query_label)


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
