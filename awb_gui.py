import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import ToastNotification

# ========== 1. 核心算法 ==========
def gray_world_awb(img, use_optimized=False):
    """
    Gray World White Balance algorithm.
    img: BGR image from OpenCV
    """
    b, g, r = cv2.split(img)
    
    if use_optimized:
        # 优化版：只考虑中间亮度区域，过滤高光和阴影的影响
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray >= 60) & (gray <= 180)
        valid_ratio = np.sum(mask)/mask.size
        if valid_ratio >= 0.1:
            b_avg = np.mean(b[mask])
            g_avg = np.mean(g[mask])
            r_avg = np.mean(r[mask])
        else:
            b_avg = np.mean(b)
            g_avg = np.mean(g)
            r_avg = np.mean(r)
    else:
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)
    
    K = (b_avg + g_avg + r_avg) / 3.0
    gain_b = K / (b_avg + 1e-6)
    gain_g = K / (g_avg + 1e-6)
    gain_r = K / (r_avg + 1e-6)
    
    b_new = np.clip(b.astype(float) * gain_b, 0, 255)
    g_new = np.clip(g.astype(float) * gain_g, 0, 255)
    r_new = np.clip(r.astype(float) * gain_r, 0, 255)
    
    return cv2.merge([b_new.astype(np.uint8), g_new.astype(np.uint8), r_new.astype(np.uint8)])

# ========== 2. 现代 GUI 设计 ==========
class AWBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity AWB - 智能白平衡校正")
        self.root.geometry("1400x900")
        
        # 状态变量
        self.original_img = None
        self.processed_img = None
        self.current_path = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- 侧边栏 ---
        self.sidebar = ttk.Frame(self.root, bootstyle=DARK, width=300, padding=20)
        self.sidebar.pack(side=LEFT, fill=Y)
        
        # 侧边栏标题
        ttk.Label(self.sidebar, text="🎨 AWB 工具箱", font=("Segoe UI Variable Display Semibold", 18), 
                  bootstyle=(INVERSE, DARK)).pack(pady=(10, 30))
        
        # 操作容器
        controls_frame = ttk.Frame(self.sidebar, bootstyle=DARK)
        controls_frame.pack(fill=X)
        
        # 按钮
        self.btn_open = ttk.Button(controls_frame, text="📷 导入图片", command=self.open_image, 
                                   bootstyle=LIGHT, width=20)
        self.btn_open.pack(pady=10)
        
        ttk.Separator(controls_frame, bootstyle=SECONDARY).pack(fill=X, pady=20)
        
        ttk.Label(controls_frame, text="算法选择", font=("微软雅黑", 10), bootstyle=(INVERSE, DARK)).pack(anchor=W)
        self.algo_var = tk.StringVar(value="优化版算法 (v2)")
        self.algo_combo = ttk.Combobox(controls_frame, textvariable=self.algo_var, 
                                       values=["基础版算法 (v0)", "优化版算法 (v2)"],
                                       state="readonly")
        self.algo_combo.pack(fill=X, pady=(5, 20))
        
        self.btn_process = ttk.Button(controls_frame, text="✨ 执行校正", command=self.process_image, 
                                      bootstyle=SUCCESS, width=20)
        self.btn_process.pack(pady=10)
        
        self.btn_save = ttk.Button(controls_frame, text="💾 导出结果", command=self.save_image, 
                                   bootstyle=(INFO, OUTLINE), width=20)
        self.btn_save.pack(pady=10)
        
        # 关于信息 (底部)
        ttk.Label(self.sidebar, text="v1.1 Premium Build\n© 2024 Antigravity", 
                  font=("Segoe UI", 8), bootstyle=(INVERSE, DARK), justify=CENTER).pack(side=BOTTOM, pady=20)
        
        # --- 主显示区 ---
        self.main_content = ttk.Frame(self.root, padding=20)
        self.main_content.pack(side=LEFT, fill=BOTH, expand=True)
        
        # 顶部状态条
        self.header_frame = ttk.Frame(self.main_content)
        self.header_frame.pack(fill=X, pady=(0, 20))
        self.status_var = tk.StringVar(value="等待导入...")
        ttk.Label(self.header_frame, textvariable=self.status_var, font=("Segoe UI", 12), 
                  bootstyle=SECONDARY).pack(side=LEFT)
        
        # 图片展示容器 (双栏布局)
        self.view_port = ttk.Frame(self.main_content)
        self.view_port.pack(fill=BOTH, expand=True)
        
        # 原图卡片
        self.left_card = ttk.Labelframe(self.view_port, text=" 原始效果 ", padding=10)
        self.left_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        self.canvas_left = tk.Canvas(self.left_card, bg="#f8f9fa", highlightthickness=0)
        self.canvas_left.pack(fill=BOTH, expand=True)
        
        # 校正图卡片
        self.right_card = ttk.Labelframe(self.view_port, text=" AWB 校正结果 ", padding=10)
        self.right_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0))
        self.canvas_right = tk.Canvas(self.right_card, bg="#f8f9fa", highlightthickness=0)
        self.canvas_right.pack(fill=BOTH, expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
            
        self.current_path = file_path
        self.original_img = cv2.imread(file_path)
        self.show_on_canvas(self.original_img, self.canvas_left)
        self.status_var.set(f"已加载: {file_path.split('/')[-1]}")
        
        # 清空右侧
        self.canvas_right.delete("all")
        self.processed_img = None

    def process_image(self):
        if self.original_img is None:
            ToastNotification(title="提示", message="请先导入一张图片", duration=3000, bootstyle=WARNING).show()
            return
            
        use_optimized = "优化版" in self.algo_var.get()
        self.processed_img = gray_world_awb(self.original_img, use_optimized=use_optimized)
        self.show_on_canvas(self.processed_img, self.canvas_right)
        self.status_var.set("校正完成！预览已就绪。")
        ToastNotification(title="成功", message="智能白平衡算法处理完毕", duration=2000, bootstyle=SUCCESS).show_toast()

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("警告", "没有可供保存的校正结果！")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                            filetypes=[("JPG Image", "*.jpg"), ("PNG Image", "*.png")])
        if path:
            cv2.imwrite(path, self.processed_img)
            ToastNotification(title="导出成功", message=f"图片已保存至该位置", duration=3000, bootstyle=INFO).show_toast()

    def show_on_canvas(self, img, canvas):
        # 转换并缩放
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 强制获取画布当前大小 (延迟一下获取更准)
        self.root.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw < 10: cw, ch = 600, 600 # 初始兜底
        
        img_pil.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(cw//2, ch//2, anchor=CENTER, image=img_tk)
        canvas.image = img_tk

if __name__ == "__main__":
    # 使用 pulse 主题，更具科技感
    root = ttk.Window(themename="pulse")
    app = AWBApp(root)
    root.mainloop()
