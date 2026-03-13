#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像匹配GUI界面
基于特征提取的图像检索系统
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
from feature_extraction import ImageMatcher
# 移除matplotlib依赖以避免NumPy兼容性问题

class ImageMatcherGUI:
    """图像匹配GUI界面类"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        self.root.title("特征提取与图像匹配系统")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.matcher = ImageMatcher()
        self.query_image = None
        self.query_image_path = None
        self.current_algorithm = tk.StringVar(value="SIFT")
        self.database_loaded = False
        
        # 创建界面
        self.create_widgets()
        
        # 加载图像数据库
        self.load_database_async()
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="特征提取与图像匹配系统", 
                               font=('宋体', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 算法选择
        ttk.Label(control_frame, text="选择算法:", font=('宋体', 10)).grid(row=0, column=0, padx=(0, 10))
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.current_algorithm,
                                     values=["SIFT", "KAZE", "ORB"], state="readonly", width=10)
        algorithm_combo.grid(row=0, column=1, padx=(0, 20))
        
        # 选择查询图像按钮
        select_btn = ttk.Button(control_frame, text="选择查询图像", 
                               command=self.select_query_image)
        select_btn.grid(row=0, column=2, padx=(0, 20))
        
        # 开始匹配按钮
        self.match_btn = ttk.Button(control_frame, text="开始匹配", 
                                   command=self.start_matching, state="disabled")
        self.match_btn.grid(row=0, column=3, padx=(0, 20))
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="正在加载图像数据库...", 
                                     font=('宋体', 10))
        self.status_label.grid(row=0, column=4, padx=(20, 0))
        
        # 图像显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # 查询图像显示
        query_frame = ttk.LabelFrame(image_frame, text="查询图像", padding="10")
        query_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.query_canvas = tk.Canvas(query_frame, width=300, height=300, bg="white")
        self.query_canvas.pack(expand=True, fill=tk.BOTH)
        
        # 匹配结果显示
        result_frame = ttk.LabelFrame(image_frame, text="匹配结果", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.result_canvas = tk.Canvas(result_frame, width=300, height=300, bg="white")
        self.result_canvas.pack(expand=True, fill=tk.BOTH)
        
        # 结果信息显示
        info_frame = ttk.LabelFrame(main_frame, text="匹配信息", padding="10")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 创建文本框显示详细信息
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, font=('宋体', 9))
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_database_async(self):
        """异步加载图像数据库"""
        def load_database():
            try:
                image_folder = "/Users/tuxol/Documents/DataVault/CST/Visual Computing/Lab1/image"
                self.matcher.load_image_database(image_folder)
                self.database_loaded = True
                
                # 更新UI
                self.root.after(0, self.on_database_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_database_error(str(e)))
        
        # 在后台线程中加载数据库
        threading.Thread(target=load_database, daemon=True).start()
    
    def on_database_loaded(self):
        """数据库加载完成回调"""
        self.status_label.config(text=f"数据库已加载 ({len(self.matcher.image_database)} 张图像)")
        self.update_match_button_state()
    
    def on_database_error(self, error_msg):
        """数据库加载错误回调"""
        self.status_label.config(text="数据库加载失败")
        messagebox.showerror("错误", f"加载图像数据库失败: {error_msg}")
    
    def select_query_image(self):
        """选择查询图像"""
        file_path = filedialog.askopenfilename(
            title="选择查询图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # 加载图像
                self.query_image = cv2.imread(file_path)
                self.query_image_path = file_path
                
                # 显示图像
                self.display_image(self.query_image, self.query_canvas)
                
                # 更新状态
                filename = os.path.basename(file_path)
                self.status_label.config(text=f"已选择查询图像: {filename}")
                
                # 更新按钮状态
                self.update_match_button_state()
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {str(e)}")
    
    def update_match_button_state(self):
        """更新匹配按钮状态"""
        if self.database_loaded and self.query_image is not None:
            self.match_btn.config(state="normal")
        else:
            self.match_btn.config(state="disabled")
    
    def display_image(self, cv_image, canvas):
        """在画布上显示图像"""
        # 转换颜色空间
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image
        
        # 获取画布尺寸
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 300, 300
        
        # 调整图像尺寸
        pil_image = Image.fromarray(rgb_image)
        pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter格式
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示图像
        canvas.delete("all")
        x = (canvas_width - pil_image.width) // 2
        y = (canvas_height - pil_image.height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
        
        # 保存引用防止被垃圾回收
        canvas.image = tk_image
    
    def start_matching(self):
        """开始匹配"""
        if not self.database_loaded or self.query_image is None:
            return
        
        # 禁用按钮
        self.match_btn.config(state="disabled")
        self.status_label.config(text="正在匹配...")
        
        # 清空信息显示
        self.info_text.delete(1.0, tk.END)
        
        # 在后台线程中进行匹配
        def match_images():
            try:
                algorithm = self.current_algorithm.get()
                result = self.matcher.find_best_match(self.query_image, algorithm)
                
                # 更新UI
                self.root.after(0, lambda: self.on_matching_complete(result, algorithm))
            except Exception as e:
                self.root.after(0, lambda: self.on_matching_error(str(e)))
        
        threading.Thread(target=match_images, daemon=True).start()
    
    def on_matching_complete(self, result, algorithm):
        """匹配完成回调"""
        self.match_btn.config(state="normal")
        
        if result:
            # 显示匹配结果图像
            self.display_image(result['image'], self.result_canvas)
            
            # 显示匹配信息
            info_text = f"算法: {algorithm}\n"
            info_text += f"最佳匹配图像: {os.path.basename(result['image_path'])}\n"
            info_text += f"图像类别: {result['category']}\n"
            info_text += f"匹配点数: {result['num_matches']}\n"
            info_text += f"查询图像关键点数: {result['query_keypoints_num']}\n"
            info_text += f"特征提取时间: {result['query_extraction_time']:.4f} 秒\n"
            info_text += f"匹配时间: {result['matching_time']:.4f} 秒\n\n"
            
            # 显示前10个匹配结果
            info_text += "前10个匹配结果:\n"
            for i, match_result in enumerate(result['all_results'][:10]):
                info_text += f"{i+1}. {os.path.basename(match_result['image_path'])} "
                info_text += f"(类别: {match_result['category']}, 匹配点: {match_result['num_matches']})\n"
            
            self.info_text.insert(tk.END, info_text)
            self.status_label.config(text=f"匹配完成 - 找到 {result['num_matches']} 个匹配点")
            
        else:
            self.info_text.insert(tk.END, f"使用 {algorithm} 算法未找到匹配结果")
            self.status_label.config(text="匹配完成 - 未找到匹配结果")
    
    def on_matching_error(self, error_msg):
        """匹配错误回调"""
        self.match_btn.config(state="normal")
        self.status_label.config(text="匹配失败")
        messagebox.showerror("错误", f"图像匹配失败: {error_msg}")

def main():
    """主函数"""
    root = tk.Tk()
    app = ImageMatcherGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
