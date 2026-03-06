import cv2
import numpy as np
import customtkinter as ctk
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# 设置主题和外观
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

from core.awb_algorithms import gray_world_awb, perfect_reflector_awb

# ========== 2. GUI主程序 (CustomTkinter版) ==========

class AWBApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AWB 智能白平衡校正 Pro")
        self.geometry("1400x900")
        
        # 状态变量
        self.original_img = None
        self.processed_img = None
        self.display_original = None
        self.display_processed = None
        
        self.setup_ui()

    def setup_ui(self):
        # 配置网格布局 (1x2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 侧边栏 ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="🎨 AWB 工具箱", font=ctk.CTkFont(family="微软雅黑", size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 40))

        # 导入按钮
        self.btn_open = ctk.CTkButton(self.sidebar_frame, text="📷 导入图片", command=self.open_image, font=ctk.CTkFont(family="微软雅黑", size=14))
        self.btn_open.grid(row=1, column=0, padx=20, pady=10)

        # 算法选择标题
        self.algo_label = ctk.CTkLabel(self.sidebar_frame, text="算法选择", anchor="w", font=ctk.CTkFont(family="微软雅黑", size=12))
        self.algo_label.grid(row=2, column=0, padx=20, pady=(20, 0), sticky="w")

        # 算法下拉菜单
        self.algo_var = ctk.StringVar(value="优化版灰度世界 (v2)")
        self.algo_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["基础版灰度世界 (v0)", "优化版灰度世界 (v2)", "完美反射体算法 (wp)"],
                                         command=self.toggle_percent_param, variable=self.algo_var, font=ctk.CTkFont(family="微软雅黑", size=13))
        self.algo_menu.grid(row=3, column=0, padx=20, pady=10)

        # 完美反射体参数调节 (初始隐藏)
        self.percent_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.percent_label_title = ctk.CTkLabel(self.percent_frame, text="高光像素占比: 5.0%", font=ctk.CTkFont(family="微软雅黑", size=12))
        self.percent_label_title.pack(anchor="w", padx=0, pady=(10, 0))
        
        self.top_percent_var = ctk.DoubleVar(value=5.0)
        self.percent_slider = ctk.CTkSlider(self.percent_frame, from_=0.1, to=10.0, number_of_steps=99,
                                           variable=self.top_percent_var, command=self.update_slider_label)
        self.percent_slider.pack(fill="x", padx=0, pady=10)
        
        # 处理与保存
        self.btn_process = ctk.CTkButton(self.sidebar_frame, text="✨ 执行校正", command=self.process_image,
                                        fg_color="#3498db", hover_color="#2980b9", font=ctk.CTkFont(family="微软雅黑", size=14, weight="bold"))
        self.btn_process.grid(row=5, column=0, padx=20, pady=(30, 10))

        self.btn_save = ctk.CTkButton(self.sidebar_frame, text="💾 导出结果", command=self.save_image,
                                     fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), font=ctk.CTkFont(family="微软雅黑", size=14))
        self.btn_save.grid(row=6, column=0, padx=20, pady=10, sticky="n")

        # 版权信息
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="v2.0 \n© 2024 AWB Project", font=ctk.CTkFont(size=10))
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=20)

        # --- 主内容区 ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # 顶部状态栏
        self.status_label = ctk.CTkLabel(self.main_frame, text="等待导入图片...", font=ctk.CTkFont(family="微软雅黑", size=16))
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="w")

        # 图片区域容器
        # 原图卡片
        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.grid(row=1, column=0, padx=(0, 10), sticky="nsew")
        ctk.CTkLabel(self.left_panel, text="原始图像", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.canvas_left = ctk.CTkLabel(self.left_panel, text="尚未导入图片", text_color="gray50")
        self.canvas_left.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 处理后卡片
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.grid(row=1, column=1, padx=(10, 0), sticky="nsew")
        ctk.CTkLabel(self.right_panel, text="校正结果", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.canvas_right = ctk.CTkLabel(self.right_panel, text="等待校正...", text_color="gray50")
        self.canvas_right.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def update_slider_label(self, value):
        self.percent_label_title.configure(text=f"高光像素占比: {value:.1f}%")

    def toggle_percent_param(self, choice):
        if "完美反射体" in choice:
            self.percent_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
        else:
            self.percent_frame.grid_forget()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            # 支持中文路径的读取
            img_array = np.fromfile(file_path, dtype=np.uint8)
            self.original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if self.original_img is not None:
                self.status_label.configure(text=f"已导入: {os.path.basename(file_path)}")
                self.show_on_label(self.original_img, self.canvas_left)
                # 清除旧结果
                self.canvas_right.configure(image=None, text="等待校正...")
                self.processed_img = None
            else:
                messagebox.showerror("错误", "无法解码图片！")

    def process_image(self):
        if self.original_img is None:
            messagebox.showwarning("提示", "请先导入图片！")
            return
        
        algo = self.algo_var.get()
        self.status_label.configure(text="正在处理...")
        self.update()

        try:
            if "基础版灰度世界" in algo:
                self.processed_img = gray_world_awb(self.original_img, use_optimized=False)
            elif "优化版灰度世界" in algo:
                self.processed_img = gray_world_awb(self.original_img, use_optimized=True)
            elif "完美反射体" in algo:
                self.processed_img = perfect_reflector_awb(self.original_img, top_percent=self.top_percent_var.get()/100.0)
            
            self.show_on_label(self.processed_img, self.canvas_right)
            self.status_label.configure(text="校正完成！")
        except Exception as e:
            messagebox.showerror("处理失败", str(e))
            self.status_label.configure(text="处理出错")

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("提示", "没有可保存的校正结果！")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
        )
        if file_path:
            # 支持中文路径保存
            _, ext = os.path.splitext(file_path)
            success, encoded_img = cv2.imencode(ext, self.processed_img)
            if success:
                encoded_img.tofile(file_path)
                messagebox.showinfo("成功", "图片已保存！")
            else:
                messagebox.showerror("错误", "保存失败！")

    def show_on_label(self, img, label_widget):
        # 转换 BGR 为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取 Label 的尺寸进行缩放
        # 注意：CTkLabel 的尺寸在初次渲染前可能不准确，这里使用一个合理的默认值或获取 master 尺寸
        width = label_widget.winfo_width()
        height = label_widget.winfo_height()
        
        if width <= 1 or height <= 1: # 还没渲染
            width, height = 600, 600
        
        # OpenCV 缩放
        h, w = img_rgb.shape[:2]
        ratio = min(width/w, height/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 CTKImage
        img_pil = Image.fromarray(img_resized)
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(new_w, new_h))
        
        label_widget.configure(image=ctk_img, text="")
        # 必须保持引用
        if label_widget == self.canvas_left:
            self.display_original = ctk_img
        else:
            self.display_processed = ctk_img

if __name__ == "__main__":
    app = AWBApp()
    app.mainloop()