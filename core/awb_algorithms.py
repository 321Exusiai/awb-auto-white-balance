import cv2
import numpy as np

def gray_world_awb(img, use_optimized=True, bright_protect=True):
    """
    灰度世界白平衡算法
    :param img: 输入BGR图像
    :param use_optimized: 是否使用优化版（忽略超亮区域，防止过曝影响）
    :param bright_protect: 是否保护高光区域
    :return: 校正后的BGR图像
    """
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    if use_optimized:
        if bright_protect:
            mask = (b < 250) & (g < 250) & (r < 250)
        else:
            mask = np.ones_like(b, dtype=bool)
        b_mean = np.mean(b[mask])
        g_mean = np.mean(g[mask])
        r_mean = np.mean(r[mask])
    else:
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)

    k = (r_mean + g_mean + b_mean) / 3.0
    kr = k / r_mean if r_mean > 0 else 1.0
    kg = k / g_mean if g_mean > 0 else 1.0
    kb = k / b_mean if b_mean > 0 else 1.0

    r_corrected = np.clip(r * kr, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g * kg, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b * kb, 0, 255).astype(np.uint8)

    return cv2.merge((b_corrected, g_corrected, r_corrected))


def perfect_reflector_awb(img, top_percent=5.0):
    """
    完美反射体（亮斑）白平衡算法
    :param img: 输入BGR图像
    :param top_percent: 取亮度最高的前N%作为参考白点
    :return: 校正后的BGR图像
    """
    top_percent = max(0.01, min(100.0, float(top_percent)))
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    brightness = r + g + b
    pixels_brightness = brightness.flatten()
    num_pixels = int(len(pixels_brightness) * (top_percent / 100.0))
    num_pixels = max(num_pixels, 1)

    idx = np.argsort(pixels_brightness)[-num_pixels:]
    b_white = np.mean(b.flatten()[idx])
    g_white = np.mean(g.flatten()[idx])
    r_white = np.mean(r.flatten()[idx])

    max_white = max(r_white, g_white, b_white)
    if max_white <= 0:
        return img.copy()

    scale_r = max_white / r_white if r_white > 0 else 1.0
    scale_g = max_white / g_white if g_white > 0 else 1.0
    scale_b = max_white / b_white if b_white > 0 else 1.0

    r_corrected = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b * scale_b, 0, 255).astype(np.uint8)

    return cv2.merge((b_corrected, g_corrected, r_corrected))


def gray_edge_awb(img, edge_threshold=50, use_optimized=True, bright_protect=True):
    """
    灰度边缘白平衡算法
    :param img: 输入BGR图像
    :param edge_threshold: 边缘强度阈值
    :param use_optimized: 是否使用优化的边缘提取
    :param bright_protect: 是否保护高光区域
    :return: 校正后的BGR图像
    """
    edge_threshold = max(0, min(100, float(edge_threshold)))
    img_float = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_optimized:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    threshold = np.percentile(grad_mag, edge_threshold)
    mask = grad_mag > threshold

    if bright_protect:
        b, g, r = cv2.split(img_float)
        brightness_mask = (b < 245) & (g < 245) & (r < 245)
        mask = mask & brightness_mask

    mask = mask.astype(bool)
    if np.sum(mask) < 10:
        mask = np.ones_like(grad_mag, dtype=bool)

    b, g, r = cv2.split(img_float)
    b_avg = np.mean(b[mask])
    g_avg = np.mean(g[mask])
    r_avg = np.mean(r[mask])

    max_avg = max(r_avg, g_avg, b_avg)
    if max_avg <= 0:
        return img.copy()

    scale_r = max_avg / r_avg if r_avg > 0 else 1.0
    scale_g = max_avg / g_avg if g_avg > 0 else 1.0
    scale_b = max_avg / b_avg if b_avg > 0 else 1.0

    r_corrected = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b * scale_b, 0, 255).astype(np.uint8)

    return cv2.merge((b_corrected, g_corrected, r_corrected))


def gamma_correction(img, gamma=1.0):
    """
    伽马校正
    :param img: 输入BGR图像
    :param gamma: 伽马值，<1变亮，>1变暗
    :return: 校正后的BGR图像
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def auto_gamma_correction(img, target_brightness=128):
    """
    自动伽马校正
    :param img: 输入BGR图像
    :param target_brightness: 目标平均亮度
    :return: (校正后的图像, 计算出的gamma值)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    if current_brightness <= 1:
        return img.copy(), 1.0
    gamma = np.log(current_brightness / 255.0) / np.log(target_brightness / 255.0)
    gamma = np.clip(gamma, 0.1, 3.0)
    return gamma_correction(img, gamma), gamma


def edge_detection_visualization(
    img, 
    edge_threshold=None,
    edge_color=(0, 50, 255),
    overlay_alpha=0.7,
    fill_area=False,
    fill_color=(0, 255, 0, 50),
    edge_thickness=2,
    blur_kernel_size=5,
    use_histogram_equalization=False
):
    """
    优化版边缘检测可视化（更智能、更精准、更醒目）
    :param img: 输入BGR图像
    :param edge_threshold: 边缘检测阈值，None时自动用Otsu法计算
    :param edge_color: 边缘的颜色，默认亮红色 (B, G, R)
    :param overlay_alpha: 原图的透明度，0.0~1.0，越小边缘越突出
    :param fill_area: 是否填充边缘围成的区域
    :param fill_color: 填充颜色，带透明度 (B, G, R, A)，A=0~255
    :param edge_thickness: 边缘加粗的厚度，1~5之间
    :param blur_kernel_size: 高斯模糊核大小（必须是奇数），3,5,7等
    :param use_histogram_equalization: 是否使用直方图均衡化（增强低对比度）
    :return: 边缘叠加后的可视化图像
    """
    # 1. 预处理：转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 直方图均衡化（可选）
    if use_histogram_equalization:
        gray = cv2.equalizeHist(gray)
    
    # 3. 高斯模糊降噪（支持自定义核大小）
    # 确保核大小是奇数
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # 4. 智能阈值处理：如果没给阈值，自动用Otsu法计算
    if edge_threshold is None:
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = int(otsu_thresh * 0.5)
        high_thresh = int(otsu_thresh * 1.5)
    else:
        low_thresh = int(edge_threshold * 0.5)
        high_thresh = int(edge_threshold * 1.5)
    
    # 5. 用Canny边缘检测（更智能、更精准）
    edge = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # 6. 边缘加粗（形态学膨胀），让边缘更醒目
    if edge_thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_thickness, edge_thickness))
        edge = cv2.dilate(edge, kernel, iterations=1)
    
    # 7. 创建边缘彩色图
    edge_colored = np.zeros_like(img)
    edge_colored[edge > 0] = edge_color
    
    # 8. 填充边缘围成的区域
    if fill_area:
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill_layer = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        cv2.drawContours(fill_layer, contours, -1, fill_color, -1)
        img_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img_with_alpha = cv2.addWeighted(img_with_alpha, 1.0, fill_layer, fill_color[3]/255.0, 0)
        edge_colored_with_alpha = cv2.cvtColor(edge_colored, cv2.COLOR_BGR2BGRA)
        overlay = cv2.addWeighted(img_with_alpha, overlay_alpha, edge_colored_with_alpha, 1 - overlay_alpha, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    else:
        overlay = cv2.addWeighted(img, overlay_alpha, edge_colored, 1 - overlay_alpha, 0)
    
    return overlay

def vignette_correction(img, vignette_strength=1.5, smoothness=0.5):
    """
    镜头渐晕校正
    :param img: 输入BGR图像
    :param vignette_strength: 渐晕强度 (1.0~2.5)
    :param smoothness: 过渡平滑度 (0.1~1.0)
    :return: 校正后的BGR图像
    """
    vignette_strength = max(1.0, float(vignette_strength))
    smoothness = max(0.01, min(1.0, float(smoothness)))
    height, width = img.shape[:2]
    # 生成从中心到边缘的距离矩阵 (范围0到1)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    radius = np.sqrt(X**2 + Y**2)
    radius = np.clip(radius, 0, 1)
    
    # 增益计算，远离中心处亮度增强
    # smoothness控制增益的空间衰减速率
    power = 1.0 / max(smoothness, 0.01)
    gain = 1 + (vignette_strength - 1.0) * (radius ** power)
    
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    b_corrected = np.clip(b * gain, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g * gain, 0, 255).astype(np.uint8)
    r_corrected = np.clip(r * gain, 0, 255).astype(np.uint8)
    
    return cv2.merge((b_corrected, g_corrected, r_corrected))

def clahe_contrast_enhancement(img, clip_limit=2.0, grid_size=(8,8)):
    """
    自适应对比度增强（CLAHE）——适配无人机水利航拍图
    :param img: 输入BGR图像（渐晕校正后的图）
    :param clip_limit: 对比度限幅（越大增强越明显，默认2.0）
    :param grid_size: 分块大小（(8,8)适合航拍图，太小会有块效应）
    :return: 增强后的BGR图像
    """
    clip_limit = max(0.1, float(clip_limit))
    # 1. 转LAB色彩空间（避免直接增强RGB导致色差）
    # LAB：L=亮度通道，A/B=色彩通道→只增强亮度，不破坏色彩
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 2. 初始化CLAHE，设置参数
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    # 只增强亮度通道（核心：色彩不变，只提对比度）
    l_channel_enhanced = clahe.apply(l_channel)
    
    # 3. 合并通道，转回BGR
    lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 4. 轻微限制像素值，防止极个别点过曝
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    
    return enhanced_img

def dehaze(img, dehaze_strength=0.9, guide_radius=60, eps=1e-3):
    """
    快速去雾（基于暗通道先验+快速引导滤波）——适配无人机水利航拍图
    :param img: 输入BGR图像（对比度增强后的图）
    :param dehaze_strength: 去雾强度（0.0~1.0，越大去雾越明显，默认0.9）
    :param guide_radius: 引导滤波半径（越大去雾越平滑，默认60）
    :param eps: 引导滤波正则化参数（默认1e-3，不用改）
    :return: 去雾后的BGR图像
    """
    dehaze_strength = max(0.0, min(1.0, float(dehaze_strength)))
    guide_radius = max(1, int(guide_radius))
    
    # 1. 转float32，避免溢出
    img_float = img.astype(np.float32) / 255.0
    h, w = img_float.shape[:2]
    
    # 2. 计算暗通道（每个像素的R/G/B最小值）
    dark_channel = np.min(img_float, axis=2)
    
    # 3. 估计大气光（取暗通道最亮的0.1%像素的平均亮度）
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))  # 取最亮的0.1%
    dark_flat = dark_channel.flatten()
    indices = np.argsort(dark_flat)[-num_brightest:]  # 最亮的像素索引
    atmospheric_light = np.mean(img_float.reshape(-1, 3)[indices], axis=0)  # 大气光
    
    # 4. 计算初始透射率
    omega = 1.0 - dehaze_strength  # 去雾强度转换
    transmission = 1.0 - omega * dark_channel  # 初始透射率
    
    # 5. 快速引导滤波（平滑透射率，避免块效应）
    # 用原图的灰度图作为引导图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = cv2.ximgproc.guidedFilter(
        guide=gray,
        src=transmission,
        radius=guide_radius,
        eps=eps
    )
    
    # 6. 限制透射率下限（避免除零）
    transmission = np.clip(transmission, 0.1, 1.0)
    
    # 7. 还原无雾图像（核心公式）
    dehazed = np.zeros_like(img_float)
    for c in range(3):
        dehazed[:, :, c] = (img_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]
    
    # 8. 限制像素值，转回uint8
    dehazed = np.clip(dehazed, 0.0, 1.0)
    dehazed = (dehazed * 255).astype(np.uint8)
    
    return dehazed

def analyze_haze_info(img, dehaze_strength=0.9):
    """
    雾信息全量分析：输出雾的颜色、浓度分布、透射率图、细节质量评级
    :param img: 输入BGR航拍原图
    :param dehaze_strength: 和你去雾功能一致的强度参数
    :return: 分析结果字典（所有可视化数据+评级）
    """
    dehaze_strength = max(0.0, min(1.0, float(dehaze_strength)))
    # --------------------------
    # 1. 复用你去雾代码的核心计算
    # --------------------------
    img_float = img.astype(np.float32) / 255.0
    h, w = img_float.shape[:2]
    
    # 计算暗通道
    dark_channel = np.min(img_float, axis=2)
    
    # 估计大气光（雾的颜色）
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))
    dark_flat = dark_channel.flatten()
    indices = np.argsort(dark_flat)[-num_brightest:]
    atmospheric_light = np.mean(img_float.reshape(-1, 3)[indices], axis=0)  # 雾的颜色（BGR，0~1）
    
    # 计算透射率（透视率）
    omega = 1.0 - dehaze_strength
    transmission = 1.0 - omega * dark_channel
    # 引导滤波平滑（和你去雾代码一致）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission_smoothed = cv2.ximgproc.guidedFilter(
        guide=gray, src=transmission, radius=60, eps=1e-3
    )
    transmission_smoothed = np.clip(transmission_smoothed, 0.1, 1.0)

    # --------------------------
    # 2. 雾的颜色处理
    # --------------------------
    # 把0~1的BGR转成0~255的RGB，方便可视化
    haze_color_bgr = (atmospheric_light * 255).astype(np.uint8)
    haze_color_rgb = haze_color_bgr[::-1]  # 转RGB给前端显示
    haze_color_hex = f"#{haze_color_rgb[0]:02x}{haze_color_rgb[1]:02x}{haze_color_rgb[2]:02x}"

    # --------------------------
    # 3. 雾的位置/浓度标注图（多级增强版）
    # --------------------------
    # 生成雾浓度热力图（蓝=无雾，红=浓雾）
    transmission_vis = (transmission_smoothed * 255).astype(np.uint8)
    haze_heatmap = cv2.applyColorMap(255 - transmission_vis, cv2.COLORMAP_JET)
    haze_overlay = cv2.addWeighted(img, 0.6, haze_heatmap, 0.4, 0)
    
    # 多级雾区标注核心逻辑
    levels = [
        {"name": "Heavy", "range": (0.0, 0.3), "color": (0, 0, 255), "label": "浓雾区"},    # 红色
        {"name": "Moderate", "range": (0.3, 0.5), "color": (0, 165, 255), "label": "中雾区"},# 橙色
        {"name": "Light", "range": (0.5, 0.8), "color": (0, 255, 255), "label": "薄雾区"}   # 黄色
    ]
    
    haze_details = []
    for lv in levels:
        mask = (transmission_smoothed >= lv["range"][0]) & (transmission_smoothed < lv["range"][1])
        area_ratio = np.sum(mask) / (h * w)
        if area_ratio > 0.05: # 只有该等级区域超过5%才标注
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(haze_overlay, contours, -1, lv["color"], 2)
            haze_details.append({"level": lv["label"], "ratio": round(area_ratio * 100, 1)})

    # 区域化分析（田字格划分）
    h_mid, w_mid = h // 2, w // 2
    quadrants = {
        "左上": transmission_smoothed[:h_mid, :w_mid],
        "右上": transmission_smoothed[:h_mid, w_mid:],
        "左下": transmission_smoothed[h_mid:, :w_mid],
        "右下": transmission_smoothed[h_mid:, w_mid:]
    }
    regional_haze = {k: round((1.0 - np.mean(v)) * 100, 1) for k, v in quadrants.items()}

    # --------------------------
    # 4. 照片细节质量评级（针对水利航拍场景优化）
    # --------------------------
    # 维度的计算逻辑保持不变，但增加多级判定
    # 4个核心评分维度（0~100分，越高质量越好）
    # 维度1：平均雾浓度（权重40%）
    avg_transmission = np.mean(transmission_smoothed)
    haze_score = np.clip(avg_transmission * 100, 0, 100)
    
    # 维度2：图像对比度（权重30%）
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray_img)  # 灰度标准差越大，对比度越高
    contrast_score = np.clip(contrast / 80 * 100, 0, 100)
    
    # 维度3：边缘清晰度（权重20%）
    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    edge_score = np.clip(edge_density * 1000, 0, 100)
    
    # 维度4：暗通道均匀性（权重10%，避免局部过曝/过暗）
    dark_std = np.std(dark_channel)
    dark_score = np.clip(100 - dark_std * 200, 0, 100)
    
    # 加权总分
    total_score = (
        haze_score * 0.4
        + contrast_score * 0.3
        + edge_score * 0.2
        + dark_score * 0.1
    )
    total_score = round(total_score, 1)
    
    # 评级（5个等级，贴合水利航拍场景）
    if total_score >= 90:
        quality_level = "优秀"
        quality_desc = "无雾，细节清晰"
    elif total_score >= 75:
        quality_level = "良好"
        quality_desc = "轻微薄雾，细节完整，可直接用于分析"
    elif total_score >= 60:
        quality_level = "一般"
        quality_desc = "中等雾感，需去雾后再做分析"
    elif total_score >= 40:
        quality_level = "较差"
        quality_desc = "浓雾，细节丢失严重，去雾后效果有限"
    else:
        quality_level = "极差"
        quality_desc = "严重雾霾，无法用于场景分析"

    # --------------------------
    # 5. 整理所有结果返回
    # --------------------------
    return {
        # 雾的核心参数
        "haze_color_bgr": haze_color_bgr,
        "haze_color_rgb": haze_color_rgb,
        "haze_color_hex": haze_color_hex,
        "avg_transmission": round(avg_transmission, 3),  # 平均透视率
        "transmission_map": (transmission_smoothed * 255).astype(np.uint8),  # 透视率灰度图
        "haze_details": haze_details,        # 多级雾区细节
        "regional_haze": regional_haze,     # 区域化分布
        # 可视化图
        "haze_heatmap_overlay": haze_overlay,  # 雾区标注叠加图
        # 质量评级
        "quality_score": total_score,
        "quality_level": quality_level,
        "quality_desc": quality_desc,
        # 分项得分
        "sub_scores": {
            "haze_score": round(haze_score, 1),
            "contrast_score": round(contrast_score, 1),
            "edge_score": round(edge_score, 1),
            "dark_score": round(dark_score, 1)
        }
    }

def get_transmission_heatmap(transmission_map):
    """把透射率灰度图转成彩色热力图，方便可视化"""
    return cv2.applyColorMap(255 - transmission_map, cv2.COLORMAP_JET)