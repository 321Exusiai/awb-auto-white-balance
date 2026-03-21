import cv2
import numpy as np
import customtkinter as ctk
import os
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import time

# 尝试开启 Windows 高 DPI 感知，解决界面模糊问题
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# 设置主题和外观
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue") # 改为更现代的蓝色主题

from core.awb_algorithms import gray_world_awb, perfect_reflector_awb, gray_edge_awb, gamma_correction, auto_gamma_correction, edge_detection_visualization, vignette_correction, clahe_contrast_enhancement, dehaze, analyze_haze_info, get_transmission_heatmap

class ZoomableCanvas(Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_id = None
        self.pil_image = None
        self.tk_image = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._drag_data = {"x": 0, "y": 0}
        self.sync_canvas = None
        self.original_width = 0                                                                                                                                 
        self.original_height = 0
        
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<Configure>", self.on_resize)

    def set_sync_canvas(self, other):
        self.sync_canvas = other

    def set_image(self, img_bgr):
        if img_bgr is None:
            self.delete("all")
            self.pil_image = None
            self.image_id = None  # 核心修复点：清空旧图片的 ID
            return
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(img_rgb)
        self.original_width, self.original_height = self.pil_image.size
        # Force a fit if dimensions are already defined, else rely on on_resize
        self.fit_to_window()
        self.redraw()

    def fit_to_window(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or h <= 1 or not self.pil_image:
            return
        
        ratio = min(w / self.original_width, h / self.original_height)
        self.scale = ratio * 0.95
        self.offset_x = w / 2
        self.offset_y = h / 2

    def redraw(self, sync=True):
        if not self.pil_image: return
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or h <= 1: return

        if self.scale < 0.01: self.scale = 0.01
        
        new_w = int(self.original_width * self.scale)
        new_h = int(self.original_height * self.scale)
        if new_w <= 0 or new_h <= 0: return
        
        resized = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        if self.image_id is None:
            self.image_id = self.create_image(self.offset_x, self.offset_y, image=self.tk_image, anchor="center")
        else:
            self.itemconfig(self.image_id, image=self.tk_image)
            self.coords(self.image_id, self.offset_x, self.offset_y)
            
        if sync and self.sync_canvas:
            self.sync_canvas.scale = self.scale
            self.sync_canvas.offset_x = self.offset_x
            self.sync_canvas.offset_y = self.offset_y
            self.sync_canvas.redraw(sync=False)

    def on_resize(self, event):
        if self.pil_image:
            if self.offset_x == 0 and self.offset_y == 0:
                self.fit_to_window()
            self.redraw()

    def zoom(self, event):
        if not self.pil_image: return
        factor = 1.1 if event.delta > 0 else 0.9
        
        x = event.x
        y = event.y
        
        self.offset_x = x + (self.offset_x - x) * factor
        self.offset_y = y + (self.offset_y - y) * factor
        self.scale *= factor
        self.redraw()

    def start_pan(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def pan(self, event):
        if not self.pil_image: return
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        self.offset_x += dx
        self.offset_y += dy
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self.redraw()

class DarkroomNonclassicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Darkroom Nonclassic")
        self.geometry("1400x900")
        self.iconbitmap(None)
        
        # 绑定关闭窗口事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 多图模型状态
        self.images = []
        self.current_index = None

        # 当前索引对应的图像缓存
        self.original_img = None
        self.processed_img = None
        self.display_original = None
        self.display_processed = None
        # 图像显示配置
        self.FIXED_DISPLAY_SIZE = (700, 900) # 稍微调大显示区域
        self.last_process_time = 0
        self.process_delay = 50 # 毫秒，用于防抖
        self._after_id = None

        self.setup_ui()

        # 键盘快捷键
        self.bind("<Left>", self.prev_image_event)
        self.bind("<Right>", self.next_image_event)

    def on_closing(self):
        """窗口关闭确认，防止意外退出丢失进度"""
        if messagebox.askyesno("退出程序", "确实要退出 Darkroom Nonclassic 吗？\n离开后，未导出的图像处理进度将丢失。"):
            self.destroy()

    def setup_ui(self):
        # 配置网格布局
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)

        # --- 侧边栏 ---
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=2, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)
        for i in range(10):
            self.sidebar_frame.grid_rowconfigure(i, weight=0)

        # Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="📷 Darkroom\nNonclassic", 
            font=ctk.CTkFont(family="微软雅黑", size=18, weight="bold"),
            text_color="#3498db"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # 顶部按钮框架 (导入 & 批量导出)
        self.top_btn_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.top_btn_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        # 导入按钮
        self.btn_open = ctk.CTkButton(
            self.top_btn_frame, 
            text="📁 导入", 
            command=self.open_images, 
            width=110,
            font=ctk.CTkFont(family="微软雅黑", size=13),
            fg_color="#2c3e50",
            hover_color="#34495e"
        )
        self.btn_open.pack(side="left", padx=(0, 5))

        # 批量导出按钮
        self.btn_batch_export = ctk.CTkButton(
            self.top_btn_frame, 
            text="📦 批量导出", 
            command=self.batch_export, 
            width=110,
            font=ctk.CTkFont(family="微软雅黑", size=13),
            fg_color="#27ae60",
            hover_color="#2ecc71"
        )
        self.btn_batch_export.pack(side="right")

        # 算法选择
        self.algo_label = ctk.CTkLabel(self.sidebar_frame, text="算法选择", anchor="w", font=ctk.CTkFont(family="微软雅黑", size=11))
        self.algo_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        # 算法下拉菜单
        self.algo_var = ctk.StringVar(value="优化版灰度世界 (v2)")
        self.algo_menu = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=[
                "基础版灰度世界 (v0)", 
                "优化版灰度世界 (v2)", 
                "完美反射体算法 (wp)",
                "基础版灰度边缘 (e0)",
                "优化版灰度边缘 (e2)",
                "手动伽马校正 (gamma)",
                "自动伽马矫正 (auto_gamma)",
                "边缘检测可视化 (edge-vis)",
                "镜头渐晕校正 (vignette)",
                "自适应对比度增强 (CLAHE)" ,
                "快速去雾 (Dehaze)" 
            ],
            command=self.toggle_param_panel, 
            variable=self.algo_var, 
            font=ctk.CTkFont(family="微软雅黑", size=12)
        )
        self.algo_menu.grid(row=3, column=0, padx=20, pady=5)

        # 完美反射体参数调节
        self.percent_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.percent_label_title = ctk.CTkLabel(self.percent_frame, text="高光像素占比：5.0%", font=ctk.CTkFont(family="微软雅黑", size=11))
        self.percent_label_title.pack(anchor="w", padx=0, pady=(5, 0))
        
        self.top_percent_var = ctk.DoubleVar(value=5.0)
        self.percent_slider = ctk.CTkSlider(
            self.percent_frame, 
            from_=0.1, to=10.0, number_of_steps=99,
            variable=self.top_percent_var, 
            command=self.on_slider_change # 实时响应
        )
        self.percent_slider.pack(fill="x", padx=0, pady=(5, 5))
        
        self.btn_reset_percent = ctk.CTkButton(
            self.percent_frame, text="重置", font=ctk.CTkFont(size=10), height=20, width=60,
            fg_color="#34495e", command=lambda: [self.top_percent_var.set(5.0), self.on_slider_change(None)]
        )
        self.btn_reset_percent.pack(anchor="e")

        # 边缘阈值调节（包含新参数）
        self.edge_thresh_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.edge_thresh_label_title = ctk.CTkLabel(
            self.edge_thresh_frame, 
            text="边缘检测阈值：50", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.edge_thresh_label_title.pack(anchor="w", padx=0, pady=(5, 0))

        self.edge_thresh_var = ctk.IntVar(value=50)
        self.edge_thresh_slider = ctk.CTkSlider(
            self.edge_thresh_frame, 
            from_=10, to=100, number_of_steps=90,
            variable=self.edge_thresh_var, 
            command=self.on_slider_change # 实时响应
        )
        self.edge_thresh_slider.pack(fill="x", padx=0, pady=(5, 5))

        # 高斯模糊核大小
        self.blur_kernel_label = ctk.CTkLabel(
            self.edge_thresh_frame,
            text="高斯模糊核：5x5",
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.blur_kernel_label.pack(anchor="w", padx=0, pady=(5, 0))
        
        self.blur_kernel_var = ctk.IntVar(value=5)
        self.blur_kernel_menu = ctk.CTkOptionMenu(
            self.edge_thresh_frame,
            values=["3", "5", "7", "9", "11"],
            variable=self.blur_kernel_var,
            command=lambda _: self.on_slider_change(None), # 下拉菜单也实时响应
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.blur_kernel_menu.pack(fill="x", padx=0, pady=(2, 5))

        self.btn_reset_edge = ctk.CTkButton(
            self.edge_thresh_frame, text="重置", font=ctk.CTkFont(size=10), height=20, width=60,
            fg_color="#34495e", command=lambda: [self.edge_thresh_var.set(50), self.blur_kernel_var.set(5), self.hist_eq_var.set(False), self.on_slider_change(None)]
        )
        self.btn_reset_edge.pack(anchor="e")

        # 直方图均衡化开关
        self.hist_eq_var = ctk.BooleanVar(value=False)
        self.hist_eq_switch = ctk.CTkSwitch(
            self.edge_thresh_frame,
            text="直方图均衡化",
            variable=self.hist_eq_var,
            command=lambda: self.on_slider_change(None), # 实时响应
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.hist_eq_switch.pack(anchor="w", padx=0, pady=(5, 10))

        # 伽马值调节
        self.gamma_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.gamma_label_title = ctk.CTkLabel(
            self.gamma_frame, 
            text="伽马值 (γ): 1.0", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.gamma_label_title.pack(anchor="w", padx=0, pady=(5, 0))

        self.gamma_var = ctk.DoubleVar(value=1.0)
        self.gamma_slider = ctk.CTkSlider(
            self.gamma_frame, 
            from_=0.1, to=3.0, number_of_steps=29,
            variable=self.gamma_var, 
            command=self.on_slider_change # 统一实时响应
        )
        self.gamma_slider.pack(fill="x", padx=0, pady=(5, 5))
        
        self.btn_reset_gamma = ctk.CTkButton(
            self.gamma_frame, text="重置", font=ctk.CTkFont(size=10), height=20, width=60,
            fg_color="#34495e", command=lambda: [self.gamma_var.set(1.0), self.on_slider_change(None)]
        )
        self.btn_reset_gamma.pack(anchor="e")
        
        # 镜头渐晕校正参数调节面板（新增）
        self.vignette_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        # 校正强度
        self.vignette_strength_label = ctk.CTkLabel(
            self.vignette_frame, 
            text="渐晕校正强度：1.5", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.vignette_strength_label.pack(anchor="w", padx=0, pady=(5, 0))
        self.vignette_strength_var = ctk.DoubleVar(value=1.5)
        self.vignette_strength_slider = ctk.CTkSlider(
            self.vignette_frame, 
            from_=1.0, to=2.5, number_of_steps=15,
            variable=self.vignette_strength_var, 
            command=self.on_slider_change # 实时响应
        )
        self.vignette_strength_slider.pack(fill="x", padx=0, pady=(2, 5))

        # 过渡平滑度
        self.vignette_smooth_label = ctk.CTkLabel(
            self.vignette_frame, 
            text="过渡平滑度：0.5", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.vignette_smooth_label.pack(anchor="w", padx=0, pady=(5, 0))
        self.vignette_smooth_var = ctk.DoubleVar(value=0.5)
        self.vignette_smooth_slider = ctk.CTkSlider(
            self.vignette_frame, 
            from_=0.1, to=1.0, number_of_steps=9,  # 0.1~1.0，步长0.1
            variable=self.vignette_smooth_var, 
            command=self.on_slider_change # 实时响应
        )
        self.vignette_smooth_slider.pack(fill="x", padx=0, pady=(2, 5))

        self.btn_reset_vignette = ctk.CTkButton(
            self.vignette_frame, text="重置", font=ctk.CTkFont(size=10), height=20, width=60,
            fg_color="#34495e", command=lambda: [self.vignette_strength_var.set(1.5), self.vignette_smooth_var.set(0.5), self.on_slider_change(None)]
        )
        self.btn_reset_vignette.pack(anchor="e")
        
        # 自适应对比度增强参数面板
        self.clahe_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        # 对比度限幅
        self.clahe_clip_label = ctk.CTkLabel(
            self.clahe_frame, 
            text="对比度限幅：2.0", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.clahe_clip_label.pack(anchor="w", padx=0, pady=(5, 0))
        self.clahe_clip_var = ctk.DoubleVar(value=2.0)
        self.clahe_clip_slider = ctk.CTkSlider(
            self.clahe_frame, 
            from_=1.0, to=4.0, number_of_steps=30,  # 1.0~4.0，步长0.1
            variable=self.clahe_clip_var, 
            command=self.on_slider_change
        )
        self.clahe_clip_slider.pack(fill="x", padx=0, pady=(2, 5))

        # 分块大小（下拉选择，避免用户选到不合适的值）
        self.clahe_grid_label = ctk.CTkLabel(
            self.clahe_frame, 
            text="分块大小：8×8", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.clahe_grid_label.pack(anchor="w", padx=0, pady=(5, 0))
        self.clahe_grid_var = ctk.StringVar(value="8")
        self.clahe_grid_menu = ctk.CTkOptionMenu(
            self.clahe_frame,
            values=["8", "16", "24"],  # 只给安全选项，避免块效应
            variable=self.clahe_grid_var,
            command=lambda _: self.on_slider_change(None),
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.clahe_grid_menu.pack(fill="x", padx=0, pady=(2, 5))
        
        self.btn_reset_clahe = ctk.CTkButton(
            self.clahe_frame, text="重置", font=ctk.CTkFont(size=10), height=20, width=60,
            fg_color="#34495e", command=lambda: [self.clahe_clip_var.set(2.0), self.clahe_grid_var.set("8"), self.on_slider_change(None)]
        )
        self.btn_reset_clahe.pack(anchor="e")

        # 快速去雾参数面板（新增）
        self.dehaze_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        # 去雾强度
        self.dehaze_strength_label = ctk.CTkLabel(
            self.dehaze_frame, 
            text="去雾强度：0.9", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.dehaze_strength_label.pack(anchor="w", padx=0, pady=(5, 0))
        self.dehaze_strength_var = ctk.DoubleVar(value=0.9)
        self.dehaze_strength_slider = ctk.CTkSlider(
            self.dehaze_frame, 
            from_=0.5, to=1.0, number_of_steps=50,  # 0.5~1.0，步长0.01
            variable=self.dehaze_strength_var, 
            command=self.update_dehaze_strength_label
        )
        self.dehaze_strength_slider.pack(fill="x", padx=0, pady=(2, 5))

        # 引导滤波半径（下拉选择，避免用户选到不合适的值）
        self.dehaze_radius_label = ctk.CTkLabel(
            self.dehaze_frame, 
            text="引导滤波半径：60", 
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.dehaze_radius_label.pack(anchor="w", padx=0, pady=(5, 0))
        self.dehaze_radius_var = ctk.StringVar(value="60")
        self.dehaze_radius_menu = ctk.CTkOptionMenu(
            self.dehaze_frame,
            values=["40", "60", "80", "100"],  # 只给安全选项
            variable=self.dehaze_radius_var,
            command=self.update_dehaze_radius_label,
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.dehaze_radius_menu.pack(fill="x", padx=0, pady=(2, 10))
        
        # 雾信息分析面板（新增）
        self.haze_analysis_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        # 雾的颜色展示
        self.haze_color_info_frame = ctk.CTkFrame(self.haze_analysis_frame, fg_color="transparent")
        self.haze_color_info_frame.pack(fill="x", pady=(5, 0))
        
        self.haze_color_label = ctk.CTkLabel(self.haze_color_info_frame, text="雾色：", font=ctk.CTkFont(family="微软雅黑", size=11))
        self.haze_color_label.pack(side="left")
        
        self.haze_color_box = ctk.CTkFrame(self.haze_color_info_frame, width=40, height=20, fg_color="#ffffff")
        self.haze_color_box.pack(side="left", padx=5)
        
        self.haze_color_hex_label = ctk.CTkLabel(self.haze_color_info_frame, text="#FFFFFF", font=ctk.CTkFont(size=10), text_color="gray")
        self.haze_color_hex_label.pack(side="left")

        # 区域分布
        self.haze_regional_label = ctk.CTkLabel(
            self.haze_analysis_frame,
            text="区域分布：--",
            font=ctk.CTkFont(family="微软雅黑", size=11),
            justify="left",
            anchor="w"
        )
        self.haze_regional_label.pack(fill="x", padx=0, pady=(5, 0))

        # 平均透视率
        self.haze_transmission_label = ctk.CTkLabel(
            self.haze_analysis_frame,
            text="平均透视率：--",
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.haze_transmission_label.pack(anchor="w", padx=0, pady=(2, 0))
        # 质量评级
        self.haze_quality_label = ctk.CTkLabel(
            self.haze_analysis_frame,
            text="图像质量：--",
            font=ctk.CTkFont(family="微软雅黑", size=11, weight="bold")
        )
        self.haze_quality_label.pack(anchor="w", padx=0, pady=(5, 0))
        # 质量描述
        self.haze_quality_desc = ctk.CTkLabel(
            self.haze_analysis_frame,
            text="",
            font=ctk.CTkFont(family="微软雅黑", size=9),
            text_color="gray"
        )
        self.haze_quality_desc.pack(anchor="w", padx=0, pady=(2, 0))
        # 分析按钮
        self.btn_analyze_haze = ctk.CTkButton(
            self.haze_analysis_frame,
            text="🔍 分析雾信息",
            command=self.analyze_current_haze,
            font=ctk.CTkFont(family="微软雅黑", size=11),
            height=25
        )
        self.btn_analyze_haze.pack(fill="x", padx=0, pady=(10, 5))
        # 查看雾区热力图按钮
        self.btn_show_haze_map = ctk.CTkButton(
            self.haze_analysis_frame,
            text="🌡️ 查看分布热力图",
            command=self.show_haze_heatmap,
            font=ctk.CTkFont(family="微软雅黑", size=11),
            height=25,
            fg_color="#d35400",
            hover_color="#e67e22"
        )
        self.btn_show_haze_map.pack(fill="x", padx=0, pady=(2, 5))
        
        self.btn_restore_view = ctk.CTkButton(
            self.haze_analysis_frame,
            text="🔄 切换回结果图",
            command=self.restore_processed_view,
            font=ctk.CTkFont(family="微软雅黑", size=11),
            height=25,
            fg_color="transparent",
            border_width=1
        )
        self.btn_restore_view.pack(fill="x", padx=0, pady=(2, 5))

        # 处理与保存
        self.btn_process = ctk.CTkButton(
            self.sidebar_frame, 
            text="✨ 执行校正", 
            command=self.process_image,
            fg_color="#3498db", 
            hover_color="#2980b9", 
            font=ctk.CTkFont(family="微软雅黑", size=13, weight="bold")
        )
        self.btn_process.grid(row=6, column=0, padx=20, pady=(15, 5))

        self.btn_save = ctk.CTkButton(
            self.sidebar_frame, 
            text="💾 导出结果", 
            command=self.save_image,
            fg_color="transparent", 
            border_width=2, 
            text_color=("gray10", "#DCE4EE"), 
            font=ctk.CTkFont(family="微软雅黑", size=13)
        )
        self.btn_save.grid(row=7, column=0, padx=20, pady=5, sticky="n")

        # 全局重置按钮
        self.btn_reset_all = ctk.CTkButton(
            self.sidebar_frame, text="🔄 重置所有参数", 
            command=self.reset_all_params,
            fg_color="transparent", border_width=1, border_color="#e74c3c", text_color="#e74c3c",
            font=ctk.CTkFont(family="微软雅黑", size=11)
        )
        self.btn_reset_all.grid(row=8, column=0, padx=20, pady=(10, 5))

        # 版权信息
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="v2.0 \n© 2024 AWB Project", font=ctk.CTkFont(size=9))
        self.appearance_mode_label.grid(row=9, column=0, padx=20, pady=(10, 5))

        # AI 模块占位区域
        self.ai_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.ai_frame.grid(row=10, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.ai_label = ctk.CTkLabel(self.ai_frame, text="AI 模块（预留）", font=ctk.CTkFont(size=10), text_color="gray")
        self.ai_label.pack(anchor="w", pady=(0, 5))
        self.ai_button = ctk.CTkButton(
            self.ai_frame,
            text="AI 修复（开发中）",
            command=self.run_ai_module_placeholder,
            state="disabled",
        )
        self.ai_button.pack(fill="x")

        # 顶部状态栏
        self.status_frame = ctk.CTkFrame(self, height=30, fg_color="#1a1a1a")
        self.status_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.status_label = ctk.CTkLabel(
            self.status_frame, 
            text="等待导入图片...", 
            font=ctk.CTkFont(family="微软雅黑", size=12),
            text_color="#95a5a6"
        )
        self.status_label.grid(row=0, column=0, padx=20, pady=5, sticky="w")

        # 导航按钮
        self.nav_frame = ctk.CTkFrame(self, height=50, fg_color="#242424")
        self.nav_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.nav_frame.grid_columnconfigure(0, weight=1)
        self.nav_frame.grid_columnconfigure(1, weight=1)

        self.view_mode = "library"
        self.btn_view_library = ctk.CTkButton(
            self.nav_frame, 
            text="📚 图库", 
            command=self.show_library_view, 
            font=ctk.CTkFont(family="微软雅黑", size=14, weight="normal"),
            fg_color="#1f6aa5", 
            hover_color="#2a7bc5",
            corner_radius=0
        )
        self.btn_view_library.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        self.btn_view_develop = ctk.CTkButton(
            self.nav_frame, 
            text="✏️ 编辑", 
            command=self.show_develop_view, 
            font=ctk.CTkFont(family="微软雅黑", size=14, weight="normal"),
            fg_color="transparent", 
            hover_color="#2a7bc5",
            corner_radius=0
        )
        self.btn_view_develop.grid(row=0, column=1, sticky="ew", padx=0, pady=0)

        # --- 主内容区 ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # --- 页面 1：图片库 ---
        self.image_list_frame = ctk.CTkFrame(self.main_frame)
        self.image_list_frame.grid_rowconfigure(1, weight=1)
        self.image_list_frame.grid_columnconfigure(0, weight=1)

        # 图片库标题和视图切换
        self.image_list_header = ctk.CTkFrame(self.image_list_frame, fg_color="transparent")
        self.image_list_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))
        self.image_list_header.grid_columnconfigure(0, weight=1)
        self.image_list_header.grid_columnconfigure(1, weight=0)

        self.image_list_label = ctk.CTkLabel(
            self.image_list_header,
            text="图片库",
            font=ctk.CTkFont(family="微软雅黑", size=14, weight="bold"),
            anchor="w",
        )
        self.image_list_label.grid(row=0, column=0, sticky="w")

        # 视图切换按钮
        self.view_mode_var = ctk.StringVar(value="grid")
        self.view_mode_frame = ctk.CTkFrame(self.image_list_header, fg_color="transparent")
        self.view_mode_frame.grid(row=0, column=1, sticky="e")

        self.btn_grid_view = ctk.CTkButton(
            self.view_mode_frame,
            text="⊞",
            width=30,
            height=30,
            command=lambda: self.switch_view_mode("grid"),
            fg_color="#1f6aa5" if self.view_mode_var.get() == "grid" else "transparent"
        )
        self.btn_grid_view.grid(row=0, column=0, padx=2)

        self.btn_list_view = ctk.CTkButton(
            self.view_mode_frame,
            text="☰",
            width=30,
            height=30,
            command=lambda: self.switch_view_mode("list"),
            fg_color="#1f6aa5" if self.view_mode_var.get() == "list" else "transparent"
        )
        self.btn_list_view.grid(row=0, column=1, padx=2)

        # 图片预览区域
        self.image_preview_frame = ctk.CTkFrame(self.image_list_frame, fg_color="transparent")
        self.image_preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_preview_frame.grid_columnconfigure((0, 1), weight=1)
        self.image_preview_frame.grid_rowconfigure(0, weight=1)
        self.image_preview_frame.grid_remove()

        self.preview_left_panel = ctk.CTkFrame(self.image_preview_frame)
        self.preview_left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ctk.CTkLabel(self.preview_left_panel, text="原始图像", font=ctk.CTkFont(weight="bold", size=12)).pack(pady=(5, 0))
        self.preview_canvas_left = ctk.CTkLabel(self.preview_left_panel, text="", text_color="gray50")
        self.preview_canvas_left.pack(fill="both", expand=True, padx=5, pady=5)

        self.preview_right_panel = ctk.CTkFrame(self.image_preview_frame)
        self.preview_right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ctk.CTkLabel(self.preview_right_panel, text="校正结果", font=ctk.CTkFont(weight="bold", size=12)).pack(pady=(5, 0))
        self.preview_canvas_right = ctk.CTkLabel(self.preview_right_panel, text="", text_color="gray50")
        self.preview_canvas_right.pack(fill="both", expand=True, padx=5, pady=5)

        self.btn_close_preview = ctk.CTkButton(
            self.image_preview_frame,
            text="✕ 关闭预览",
            width=100,
            height=30,
            command=self.close_image_preview,
            fg_color="#e74c3c",
            hover_color="#c0392b"
        )
        self.btn_close_preview.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        # 图片显示区域
        self.image_list = ctk.CTkScrollableFrame(self.image_list_frame, fg_color="transparent")
        self.image_list.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        self.image_item_buttons = []
        self.grid_columns = 5

        # --- 页面 2：调整与历史 ---
        self.develop_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.develop_frame.grid_columnconfigure(0, weight=1)
        self.develop_frame.grid_rowconfigure(0, weight=2)
        self.develop_frame.grid_rowconfigure(1, weight=1)

        # 上半部分：原图 + 处理结果
        self.compare_panel = ctk.CTkFrame(self.develop_frame)
        self.compare_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.compare_panel.grid_columnconfigure((0, 1), weight=1)
        self.compare_panel.grid_rowconfigure(0, weight=1)

        # 左侧：原始图像
        self.left_panel = ctk.CTkFrame(self.compare_panel)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ctk.CTkLabel(self.left_panel, text="原始图像（滚动缩放/拖拽平移）", font=ctk.CTkFont(weight="bold", size=12)).pack(pady=(5, 0))
        self.canvas_left = ZoomableCanvas(self.left_panel, bg="#1a1a1a", highlightthickness=0)
        self.canvas_left.pack(fill="both", expand=True, padx=5, pady=5)

        # 右侧：校正结果
        self.right_panel = ctk.CTkFrame(self.compare_panel)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ctk.CTkLabel(self.right_panel, text="校正结果（同步缩放）", font=ctk.CTkFont(weight="bold", size=12)).pack(pady=(5, 0))
        self.canvas_right = ZoomableCanvas(self.right_panel, bg="#1a1a1a", highlightthickness=0)
        self.canvas_right.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 绑定同步效果
        self.canvas_left.set_sync_canvas(self.canvas_right)
        self.canvas_right.set_sync_canvas(self.canvas_left)

        # 下半部分：历史记录面板
        self.history_frame = ctk.CTkFrame(self.develop_frame)
        self.history_frame.grid(row=1, column=0, sticky="nsew")
        self.history_frame.grid_rowconfigure(1, weight=1)
        self.history_frame.grid_columnconfigure(0, weight=1)

        self.history_label = ctk.CTkLabel(
            self.history_frame,
            text="历史记录（每张图片独立）",
            font=ctk.CTkFont(family="微软雅黑", size=13, weight="bold"),
            anchor="w",
        )
        self.history_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="ew")

        self.history_list = ctk.CTkScrollableFrame(self.history_frame, fg_color="transparent")
        self.history_list.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # 默认显示图片库视图
        self.show_library_view()

    def update_slider_label(self, value):
        self.percent_label_title.configure(text=f"高光像素占比：{value:.1f}%")

    def update_edge_thresh_label(self, value):
        self.edge_thresh_label_title.configure(text=f"边缘检测阈值：{int(value)}")

    def update_blur_kernel_label(self, value):
        self.blur_kernel_label.configure(text=f"高斯模糊核：{value}x{value}")

    # 更新渐晕校正强度标签
    def update_vignette_strength_label(self, value):
        self.vignette_strength_label.configure(text=f"渐晕校正强度：{float(value):.1f}")

    # 更新过渡平滑度标签
    def update_vignette_smooth_label(self, value):
        self.vignette_smooth_label.configure(text=f"过渡平滑度：{float(value):.1f}")

    # 更新对比度限幅标签
    def update_clahe_clip_label(self, value):
        self.clahe_clip_label.configure(text=f"对比度限幅：{float(value):.1f}")

    # 更新分块大小标签
    def update_clahe_grid_label(self, value):
        self.clahe_grid_label.configure(text=f"分块大小：{value}×{value}")

    # 更新去雾强度标签
    def update_dehaze_strength_label(self, value):
        self.dehaze_strength_label.configure(text=f"去雾强度：{float(value):.2f}")
        # 新增：调节滑块时自动更新雾分析（防抖/延迟处理）
        if self.current_index is not None and self.images:
            if hasattr(self, "_dehaze_after_id") and self._dehaze_after_id:
                self.after_cancel(self._dehaze_after_id)
            self._dehaze_after_id = self.after(200, self.analyze_current_haze)

    # 更新引导滤波半径标签
    def update_dehaze_radius_label(self, value):
        self.dehaze_radius_label.configure(text=f"引导滤波半径：{value}")

    def analyze_current_haze(self):
        """分析当前图片的雾信息"""
        if self.current_index is None or not self.images:
            return
        
        try:
            img_state = self.images[self.current_index]
            self.original_img = img_state["original"]
            dehaze_strength = self.dehaze_strength_var.get()
            
            # 调用分析函数
            self.haze_analysis_result = analyze_haze_info(
                self.original_img,
                dehaze_strength=dehaze_strength
            )
            
            # 更新界面显示
            result = self.haze_analysis_result
            # 雾的颜色
            self.haze_color_box.configure(fg_color=result["haze_color_hex"])
            self.haze_color_hex_label.configure(text=result["haze_color_hex"])
            
            # 平均透视率
            self.haze_transmission_label.configure(
                text=f"平均透视率：{result['avg_transmission']}（越低越浓）"
            )
            
            # 区域分布
            reg = result["regional_haze"]
            reg_text = f"分布：左上{reg['左上']}% | 右上{reg['右上']}%\n         左下{reg['左下']}% | 右下{reg['右下']}%"
            self.haze_regional_label.configure(text=reg_text)

            # 质量评级
            self.haze_quality_label.configure(
                text=f"图像质量：{result['quality_level']}（{result['quality_score']}分）"
            )
            self.haze_quality_desc.configure(text=result["quality_desc"])
            
            self.status_label.configure(text="雾信息全面分析完成！")
        except Exception as e:
            self.status_label.configure(text=f"分析出错: {str(e)}")

    def show_haze_heatmap(self):
        """在右侧画布显示多级雾区分布图"""
        if not hasattr(self, "haze_analysis_result"):
            self.analyze_current_haze() # 自动先分析
            if not hasattr(self, "haze_analysis_result"): return
        
        # 把雾区叠加图显示到右侧画布
        haze_overlay = self.haze_analysis_result["haze_heatmap_overlay"]
        self.show_on_label(haze_overlay, self.canvas_right)
        self.status_label.configure(text="已切换到雾分布热力图（多级颜色标注）")

    def restore_processed_view(self):
        """切换回校正后的结果图"""
        if self.processed_img is not None:
            self.show_on_label(self.processed_img, self.canvas_right)
            self.status_label.configure(text="已预览校正结果图")
        else:
            messagebox.showinfo("提示", "请先执行校正以生成结果图")

    def on_slider_change(self, value):
        """滑块变动时的统调函数（带防抖）"""
        self.show_develop_view() # 自动切换到编辑视图
        
        # 更新所有标签文字
        self.update_slider_label(self.top_percent_var.get())
        self.update_edge_thresh_label(self.edge_thresh_var.get())
        self.update_blur_kernel_label(self.blur_kernel_var.get())
        self.gamma_label_title.configure(text=f"伽马值 (γ): {self.gamma_var.get():.1f}")
        self.update_vignette_strength_label(self.vignette_strength_var.get())
        self.update_vignette_smooth_label(self.vignette_smooth_var.get())
        self.update_clahe_clip_label(self.clahe_clip_var.get())
        self.update_clahe_grid_label(self.clahe_grid_var.get())

        # 防抖处理
        if self._after_id:
            self.after_cancel(self._after_id)
        # 滑块实时预览时不保存历史，防止历史记录堆叠
        self._after_id = self.after(self.process_delay, lambda: self.process_image(save_history=False))

    def toggle_param_panel(self, choice):
    # 隐藏所有参数面板
        self.percent_frame.grid_forget()
        self.edge_thresh_frame.grid_forget()
        self.gamma_frame.grid_forget()
        self.vignette_frame.grid_forget()
        self.clahe_frame.grid_forget() 
        self.dehaze_frame.grid_forget()
        self.haze_analysis_frame.grid_forget()

        # 显示对应面板
        if "完美反射体" in choice:
            self.percent_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
        elif "灰度边缘" in choice or "边缘检测可视化" in choice:
            self.edge_thresh_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
        elif "手动伽马校正" in choice:
            self.gamma_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
        elif "镜头渐晕校正" in choice:
            self.vignette_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
        elif "自适应对比度增强" in choice: 
            self.clahe_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
        elif "快速去雾" in choice:
            self.dehaze_frame.grid(row=4, column=0, padx=20, pady=0, sticky="ew")
            self.haze_analysis_frame.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="ew")

    def reset_all_params(self):
        """重置所有参数到默认值"""
        self.top_percent_var.set(5.0)
        self.edge_thresh_var.set(50)
        self.blur_kernel_var.set(5)
        self.hist_eq_var.set(False)
        self.gamma_var.set(1.0)
        self.vignette_strength_var.set(1.5)
        self.vignette_smooth_var.set(0.5)
        self.on_slider_change(None)
        self.status_label.configure(text="所有参数已重置")

    def update_gamma_and_process(self, value):
        self.gamma_label_title.configure(text=f"伽马值 (γ): {value:.1f}")
    
        if self.current_index is None or not self.images:
            return

        img_state = self.images[self.current_index]
        self.original_img = img_state["original"]

        processed = gamma_correction(
            self.original_img,
            gamma=value
        )
        self.processed_img = processed
        img_state["processed"] = processed
        img_state.setdefault("params", {})["gamma"] = float(value)
        img_state.setdefault("history", []).append(
            {
                "type": "gamma_preview",
                "gamma": float(value),
                "image": processed.copy(),
            }
        )

        self.show_on_label(self.processed_img, self.canvas_right)
        self.status_label.configure(text=f"实时伽马校正：γ={value:.1f}")

    def open_images(self):
        file_paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_paths:
            return

        added_any = False
        for file_path in file_paths:
            if not file_path:
                continue
            img_array = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                messagebox.showerror("错误", f"无法解码图片：{os.path.basename(file_path)}")
                continue

            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                thumb_max = 80
                ratio = min(thumb_max / max(w, 1), thumb_max / max(h, 1))
                thumb_w = int(w * ratio)
                thumb_h = int(h * ratio)
                thumb_resized = cv2.resize(img_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                
                thumb_pil = Image.fromarray(thumb_resized)
                thumb_ctk = ctk.CTkImage(light_image=thumb_pil, dark_image=thumb_pil, size=(thumb_w, thumb_h))
            except Exception as e:
                messagebox.showwarning("警告", f"缩略图生成失败：{os.path.basename(file_path)}\n{str(e)}")
                continue

            self.images.append({
                "path": file_path,
                "name": os.path.basename(file_path),
                "original": img,
                "processed": None,
                "algo": None,
                "params": {},
                "history": [],
                "thumb": thumb_ctk
            })
            added_any = True

        if added_any:
            if self.current_index is None:
                self.current_index = 0
            self.refresh_image_list()
            self.select_image(self.current_index)
            self.status_label.configure(text=f"已导入 {len(self.images)} 张图片")

    def refresh_history_panel(self):
        """刷新当前图片的历史记录列表"""
        for child in self.history_list.winfo_children():
            child.destroy()

        if self.current_index is None or not self.images:
            return

        img_state = self.images[self.current_index]
        history = img_state.get("history", [])
        if not history:
            empty_label = ctk.CTkLabel(self.history_list, text="暂无历史记录", text_color="gray60")
            empty_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            return

        for idx, entry in enumerate(history):
            entry_type = entry.get("type", "process")
            algo = entry.get("algo", "")
            desc = ""
            if entry_type == "process":
                desc = f"{idx + 1}. 算法：{algo}"
                params = entry.get("params", {})
                if params:
                    desc += f" | 参数: {params}"
            elif entry_type == "gamma_preview":
                desc = f"{idx + 1}. 伽马预览 γ={entry.get('gamma', 1.0):.2f}"
            else:
                desc = f"{idx + 1}. {entry_type}"

            row_frame = ctk.CTkFrame(self.history_list, fg_color="transparent")
            row_frame.grid(row=idx, column=0, sticky="ew", padx=5, pady=2)
            row_frame.grid_columnconfigure(0, weight=1)

            label = ctk.CTkLabel(row_frame, text=desc, anchor="w")
            label.grid(row=0, column=0, sticky="w")

            if "image" in entry:
                btn_restore = ctk.CTkButton(
                    row_frame,
                    text="回退到此版本",
                    width=100,
                    command=lambda i=idx: self.restore_history_entry(i),
                )
                btn_restore.grid(row=0, column=1, padx=(10, 0))

    def restore_history_entry(self, index: int):
        if self.current_index is None or not self.images:
            return
        img_state = self.images[self.current_index]
        history = img_state.get("history", [])
        if index < 0 or index >= len(history):
            return

        entry = history[index]
        if "image" not in entry:
            return

        restored = entry["image"].copy()
        self.processed_img = restored
        img_state["processed"] = restored

        img_state.setdefault("history", []).append(
            {
                "type": "restore",
                "from_index": index,
                "image": restored.copy(),
            }
        )

        self.show_on_label(self.processed_img, self.canvas_right)
        self.status_label.configure(text=f"已回退到历史版本 #{index + 1}")
        self.refresh_history_panel()

    def run_ai_module_placeholder(self):
        messagebox.showinfo("提示", "AI 模块正在开发中，敬请期待！")

    def process_image(self, save_history=True): # 增加 save_history 参数
        if self.current_index is None or not self.images:
            messagebox.showwarning("提示", "请先导入并选择一张图片！")
            return
            
        self.show_develop_view() # 自动切换到编辑视图

        try:
            # 强化状态定位：确保处理的是当前选中的最新数据
            img_state = self.images[self.current_index]
            self.original_img = img_state["original"]
            
            algo = self.algo_var.get()
            params = {}
            
            self.status_label.configure(text=f"核心算法 {algo} 处理中，请稍候...")
            self.configure(cursor="watch") # 添加加载光标
            self.update_idletasks() # 强制界面刷新显示状态

            if "基础版灰度世界" in algo:
                processed = gray_world_awb(self.original_img, use_optimized=False)
            elif "优化版灰度世界" in algo:
                processed = gray_world_awb(self.original_img, use_optimized=True)
            elif "完美反射体" in algo:
                top_percent = self.top_percent_var.get()
                processed = perfect_reflector_awb(self.original_img, top_percent=top_percent)
                params["top_percent"] = float(self.top_percent_var.get())
            elif "基础版灰度边缘" in algo:
                edge_thresh = self.edge_thresh_var.get()
                processed = gray_edge_awb(
                    self.original_img, 
                    edge_threshold=edge_thresh,
                    use_optimized=False,
                    bright_protect=True
                )
                params["edge_threshold"] = int(edge_thresh)
            elif "优化版灰度边缘" in algo:
                edge_thresh = self.edge_thresh_var.get()
                processed = gray_edge_awb(
                    self.original_img, 
                    edge_threshold=edge_thresh,
                    use_optimized=True,
                    bright_protect=True
                )
                params["edge_threshold"] = int(edge_thresh)
            elif "手动伽马校正" in algo:
                current_gamma = self.gamma_var.get()
                processed = gamma_correction(
                    self.original_img, 
                    gamma=current_gamma
                )
                params["gamma"] = float(current_gamma)
                self.status_label.configure(text=f"手动伽马校正：γ={current_gamma:.1f}")
            elif "自动伽马矫正" in algo:
                processed, auto_gamma = auto_gamma_correction(
                    self.original_img,
                    target_brightness=128
                )
                params["auto_gamma"] = float(auto_gamma)
                self.status_label.configure(text=f"自动伽马校正：γ={auto_gamma:.2f}（自动计算）")
            elif "边缘检测可视化" in algo:
                edge_thresh = self.edge_thresh_var.get()
                blur_kernel = int(self.blur_kernel_var.get())
                use_hist_eq = self.hist_eq_var.get()
                processed = edge_detection_visualization(
                    self.original_img,
                    edge_threshold=edge_thresh,
                    edge_color=(0, 50, 255),
                    overlay_alpha=0.6,
                    edge_thickness=3,
                    fill_area=True,
                    fill_color=(0, 255, 0, 80),
                    blur_kernel_size=blur_kernel,
                    use_histogram_equalization=use_hist_eq
                )
                params["edge_threshold"] = int(edge_thresh)
                params["blur_kernel"] = int(blur_kernel)
                params["use_hist_eq"] = bool(use_hist_eq)
                self.status_label.configure(
                    text=f"边缘检测可视化：阈值={edge_thresh}，高斯核={blur_kernel}×{blur_kernel}，直方图均衡化={'开' if use_hist_eq else '关'}"
                )
            elif "镜头渐晕校正" in algo:
                vignette_strength = self.vignette_strength_var.get()
                smoothness = self.vignette_smooth_var.get()
                processed = vignette_correction(
                    self.original_img,
                    vignette_strength=vignette_strength,
                    smoothness=smoothness
                )
                params["vignette_strength"] = float(vignette_strength)
                params["smoothness"] = float(smoothness)
                self.status_label.configure(
                    text=f"镜头渐晕校正完成：强度={vignette_strength:.1f}，平滑度={smoothness:.1f}"
                )
            elif "自适应对比度增强" in algo:
                clip_limit = self.clahe_clip_var.get()
                grid_size = (int(self.clahe_grid_var.get()), int(self.clahe_grid_var.get()))
                processed = clahe_contrast_enhancement(
                    self.original_img,
                    clip_limit=clip_limit,
                    grid_size=grid_size
                )
                params["clip_limit"] = float(clip_limit)
                params["grid_size"] = grid_size
                self.status_label.configure(
                    text=f"对比度增强完成：限幅={clip_limit:.1f}，分块={grid_size[0]}×{grid_size[1]}"
                )
            elif "快速去雾" in algo:
                dehaze_strength = self.dehaze_strength_var.get()
                guide_radius = int(self.dehaze_radius_var.get())
                processed = dehaze(
                    self.original_img,
                    dehaze_strength=dehaze_strength,
                    guide_radius=guide_radius
                )
                params["dehaze_strength"] = float(dehaze_strength)
                params["guide_radius"] = guide_radius
                self.status_label.configure(
                    text=f"去雾完成：强度={dehaze_strength:.2f}，半径={guide_radius}"
                )
            else:
                messagebox.showwarning("提示", "未知算法类型")
                return

            self.processed_img = processed
            img_state["processed"] = processed
            img_state["algo"] = algo

            if save_history: # 只有显式要求时（如点击按钮）才记录历史
                img_state.setdefault("history", []).append(
                    {
                        "type": "process",
                        "algo": algo,
                        "params": dict(params),
                        "image": processed.copy(),
                    }
                )
                self.refresh_history_panel()

            # 显示结果
            self.show_on_label(self.processed_img, self.canvas_right)
            if "手动伽马校正" not in algo and "自动伽马矫正" not in algo and "边缘检测可视化" not in algo:
                self.status_label.configure(text=f"已完成：{algo} 处理成功")
            else:
                self.status_label.configure(text="预览生成完成")

            self.refresh_history_panel()
        except Exception as e:
            messagebox.showerror("处理失败", str(e))
            self.status_label.configure(text="处理出错")
        finally:
            self.configure(cursor="") # 恢复正常光标

    def batch_export(self):
        if not self.images:
            messagebox.showwarning("提示", "当前没有导入任何图片！")
            return
            
        export_count = sum(1 for img in self.images if img.get("processed") is not None)
        if export_count == 0:
            messagebox.showwarning("提示", "没有可导出的图像，请先至少校正一张图片！")
            return
            
        export_dir = filedialog.askdirectory(title="选择批量导出文件夹")
        if not export_dir:
            return
            
        # UI反聩：禁用按钮，显示进度
        self.btn_batch_export.configure(state="disabled", text="导出中...")
        self.configure(cursor="watch")
        self.status_label.configure(text=f"准备导出 {export_count} 张图片...")
        self.update_idletasks()
        
        saved = 0
        try:
            for idx, img_state in enumerate(self.images):
                if img_state.get("processed") is not None:
                    # 获取原文件名并改名
                    orig_name, ext = os.path.splitext(img_state["name"])
                    if not ext: ext = ".jpg"
                    new_name = f"{orig_name}_processed{ext}"
                    save_path = os.path.join(export_dir, new_name)
                    
                    success, encoded_img = cv2.imencode(ext, img_state["processed"])
                    if success:
                        encoded_img.tofile(save_path)
                        saved += 1
                        self.status_label.configure(text=f"正在导出: {saved}/{export_count} ({new_name})")
                        self.update_idletasks()
                        
            messagebox.showinfo("批量导出成功", f"成功导出 {saved} 张图片到：\n{export_dir}")
            self.status_label.configure(text=f"批量导出完成！共 {saved} 张。")
        except Exception as e:
            messagebox.showerror("导出错误", f"导出过程中发生错误：\n{str(e)}")
            self.status_label.configure(text="批量导出失败。")
        finally:
            self.btn_batch_export.configure(state="normal", text="📦 批量导出")
            self.configure(cursor="")

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("提示", "没有可保存的校正结果！")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
        )
        if file_path:
            _, ext = os.path.splitext(file_path)
            success, encoded_img = cv2.imencode(ext, self.processed_img)
            if success:
                encoded_img.tofile(file_path)
                messagebox.showinfo("成功", "图片已保存！")
            else:
                messagebox.showerror("错误", "保存失败！")

    def show_on_label(self, img, label_widget):
        if img is None:
            return

        if hasattr(label_widget, "set_image"):
            label_widget.set_image(img)
            # 同样保持引用，虽然不一定必要，保险起见
            if label_widget == self.canvas_left: self.display_original = label_widget.tk_image
            elif label_widget == self.canvas_right: self.display_processed = label_widget.tk_image
            return

        # 转换 BGR 到 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 获取窗口缩放比例 (DPI Scaling)
        scaling = self._get_window_scaling()
        
        # 计算显示尺寸
        max_w, max_h = self.FIXED_DISPLAY_SIZE
        w, h = pil_img.size
        
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 使用 PIL 的新算法进行高质量缩放，这比 cv2.resize 在 UI 显示上更清晰
        # CTkImage 会根据 scaling 自动处理高 DPI 渲染
        ctk_img = ctk.CTkImage(
            light_image=pil_img, 
            dark_image=pil_img, 
            size=(new_w, new_h)
        )
        
        label_widget.configure(image=ctk_img, text="")
        
        # 保持引用防止被垃圾回收
        if label_widget == self.canvas_left: self.display_original = ctk_img
        elif label_widget == self.canvas_right: self.display_processed = ctk_img
        elif label_widget == self.preview_canvas_left: self.preview_original = ctk_img
        elif label_widget == self.preview_canvas_right: self.preview_processed = ctk_img

    def switch_view_mode(self, mode):
        self.view_mode_var.set(mode)
        self.btn_grid_view.configure(fg_color="#1f6aa5" if mode == "grid" else "transparent")
        self.btn_list_view.configure(fg_color="#1f6aa5" if mode == "list" else "transparent")
        self.refresh_image_list()

    def refresh_image_list(self):
        for btn in self.image_item_buttons:
            btn.destroy()
        self.image_item_buttons.clear()

        mode = self.view_mode_var.get()
        
        if mode == "grid":
            for idx, img_state in enumerate(self.images):
                row = idx // self.grid_columns
                col = idx % self.grid_columns
                
                btn = ctk.CTkButton(
                    self.image_list,
                    text="",
                    image=img_state.get("thumb"),
                    width=120,
                    height=120,
                    command=lambda i=idx: self.select_image(i),
                )
                btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                btn.bind("<Double-1>", lambda e, i=idx: self.view_image_large(i))
                self.image_item_buttons.append(btn)
        else:
            for idx, img_state in enumerate(self.images):
                text = f"{idx + 1}. {img_state['name']}"
                btn = ctk.CTkButton(
                    self.image_list,
                    text=text,
                    anchor="w",
                    image=img_state.get("thumb"),
                    compound="left",
                    command=lambda i=idx: self.select_image(i),
                )
                btn.grid(row=idx, column=0, sticky="ew", padx=5, pady=2)
                btn.bind("<Double-1>", lambda e, i=idx: self.view_image_large(i))
                self.image_item_buttons.append(btn)

        self.highlight_current_image()

    def view_image_large(self, index):
        if not self.images or index < 0 or index >= len(self.images):
            return
        
        self.select_image(index)
        self.show_image_preview_in_library()
    
    def show_image_preview_in_library(self):
        if self.current_index is None or not self.images:
            return
        
        img_state = self.images[self.current_index]
        self.original_img = img_state["original"]
        self.processed_img = img_state.get("processed")
        
        self.image_list.grid_remove()
        self.image_preview_frame.grid()
        
        self.show_on_label(self.original_img, self.preview_canvas_left)
        
        if self.processed_img is not None:
            self.show_on_label(self.processed_img, self.preview_canvas_right)
        else:
            self.preview_canvas_right.configure(image=None, text="等待校正...")
    
    def close_image_preview(self):
        self.image_preview_frame.grid_remove()
        self.image_list.grid()

    def highlight_current_image(self):
        for idx, btn in enumerate(self.image_item_buttons):
            if idx == self.current_index:
                btn.configure(fg_color="#1f6aa5")
            else:
                btn.configure(fg_color="transparent")

    def select_image(self, index: int):
        if not self.images:
            return
        if index < 0 or index >= len(self.images):
            return

        self.current_index = index
        img_state = self.images[index]

        self.original_img = img_state["original"]
        self.processed_img = img_state.get("processed")

        self.status_label.configure(
            text=f"当前：{img_state['name']}"
        )

        self.show_on_label(self.original_img, self.canvas_left)

        # 核心修复：确保切换图片时彻底清空右侧状态
        if self.processed_img is not None:
            self.show_on_label(self.processed_img, self.canvas_right)
        else:
            self.processed_img = None
            self.display_processed = None
            if hasattr(self.canvas_right, "set_image"):
                self.canvas_right.set_image(None)
            else:
                self.canvas_right.configure(image="", text="等待校正...")
            # 状态栏明确提示
            self.status_label.configure(text=f"已切换：{img_state['name']} (待校正)")

        algo = img_state.get("algo")
        params = img_state.get("params", {})
        if algo is not None:
            self.algo_var.set(algo)
            self.toggle_param_panel(algo)

            if "完美反射体" in algo and "top_percent" in params:
                self.top_percent_var.set(params["top_percent"])
                self.update_slider_label(self.top_percent_var.get())
            if "灰度边缘" in algo and "edge_threshold" in params:
                self.edge_thresh_var.set(params["edge_threshold"])
                self.update_edge_thresh_label(self.edge_thresh_var.get())
            if "边缘检测可视化" in algo:
                if "edge_threshold" in params:
                    self.edge_thresh_var.set(params["edge_threshold"])
                    self.update_edge_thresh_label(self.edge_thresh_var.get())
                if "blur_kernel" in params:
                    self.blur_kernel_var.set(params["blur_kernel"])
                    self.update_blur_kernel_label(self.blur_kernel_var.get())
                if "use_hist_eq" in params:
                    self.hist_eq_var.set(params["use_hist_eq"])
            if "伽马" in algo and "gamma" in params:
                self.gamma_var.set(params["gamma"])
                self.gamma_label_title.configure(text=f"伽马值 (γ): {self.gamma_var.get():.1f}")
            if "镜头渐晕校正" in algo:
                if "vignette_strength" in params:
                    self.vignette_strength_var.set(params["vignette_strength"])
                    self.update_vignette_strength_label(params["vignette_strength"])
                if "smoothness" in params:
                    self.vignette_smooth_var.set(params["smoothness"])
                    self.update_vignette_smooth_label(params["smoothness"])
            if "自适应对比度增强" in algo:
                if "clip_limit" in params:
                    self.clahe_clip_var.set(params["clip_limit"])
                    self.update_clahe_clip_label(params["clip_limit"])
                if "grid_size" in params:
                    grid_val = str(params["grid_size"][0])
                    self.clahe_grid_var.set(grid_val)
                    self.update_clahe_grid_label(grid_val)
            if "快速去雾" in algo:
                if "dehaze_strength" in params:
                    self.dehaze_strength_var.set(params["dehaze_strength"])
                    self.update_dehaze_strength_label(params["dehaze_strength"])
                if "guide_radius" in params:
                    radius_val = str(params["guide_radius"])
                    self.dehaze_radius_var.set(radius_val)
                    self.update_dehaze_radius_label(radius_val)
                

        self.refresh_history_panel()
        self.highlight_current_image()

    def prev_image_event(self, event=None):
        if not self.images:
            return
        if self.current_index is None:
            self.current_index = 0
            self.select_image(self.current_index)
            return
        if self.current_index > 0:
            self.select_image(self.current_index - 1)

    def next_image_event(self, event=None):
        if not self.images:
            return
        if self.current_index is None:
            self.current_index = 0
            self.select_image(self.current_index)
            return
        if self.current_index < len(self.images) - 1:
            self.select_image(self.current_index + 1)

    def update_nav_buttons(self):
        if self.view_mode == "library":
            self.btn_view_library.configure(fg_color="#1f6aa5")
            self.btn_view_develop.configure(fg_color="transparent")
        else:
            self.btn_view_library.configure(fg_color="transparent")
            self.btn_view_develop.configure(fg_color="#1f6aa5")

    def show_library_view(self):
        self.view_mode = "library"
        self.image_list_frame.grid(row=0, column=0, sticky="nsew")
        self.develop_frame.grid_forget()
        self.update_nav_buttons()

    def show_develop_view(self):
        self.view_mode = "develop"
        self.develop_frame.grid(row=0, column=0, sticky="nsew")
        self.image_list_frame.grid_forget()
        self.update_nav_buttons()

if __name__ == "__main__":
    app = DarkroomNonclassicApp()
    app.mainloop()
