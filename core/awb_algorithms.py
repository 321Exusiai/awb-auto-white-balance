import cv2
import numpy as np

def gray_world_awb(img, use_optimized=False, bright_protect=True):
    """
    灰度世界白平衡算法（新增亮区保护）
    :param img: 输入BGR图像
    :param use_optimized: 是否使用中亮度优化版
    :param bright_protect: 是否开启高亮区域保护（解决蓝天偏色）
    """
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高亮区域掩码（亮度>200的区域不参与计算，保护蓝天/窗户）
    bright_mask = gray < 200 if bright_protect else np.ones_like(gray, dtype=bool)
    
    if use_optimized:
        # 优化版：中亮度筛选 + 亮区保护
        mid_mask = (gray >= 20) & (gray <= 240)  # 放宽亮度范围，纳入蓝天
        final_mask = mid_mask & bright_mask
        valid_ratio = np.sum(final_mask)/final_mask.size
        
        if valid_ratio >= 0.05:  # 降低有效像素阈值，适配大光比场景
            b_avg = np.mean(b[final_mask])
            g_avg = np.mean(g[final_mask])
            r_avg = np.mean(r[final_mask])
        else:
            # 有效像素不足，退化为全局计算
            final_mask = bright_mask
            b_avg = np.mean(b[final_mask])
            g_avg = np.mean(g[final_mask])
            r_avg = np.mean(r[final_mask])
    else:
        # 基础版：全局计算 + 亮区保护
        final_mask = bright_mask
        b_avg = np.mean(b[final_mask])
        g_avg = np.mean(g[final_mask])
        r_avg = np.mean(r[final_mask])
    
    # 增益计算
    mean_gray = (b_avg + g_avg + r_avg) / 3.0
    kb = mean_gray / (b_avg + 1e-6)
    kg = mean_gray / (g_avg + 1e-6)
    kr = mean_gray / (r_avg + 1e-6)
    
    # 应用增益（亮区不校正，保留原色）
    b_new = b.astype(float) * kb
    g_new = g.astype(float) * kg
    r_new = r.astype(float) * kr
    
    # 高亮区域恢复原色
    if bright_protect:
        b_new[~bright_mask] = b[~bright_mask]
        g_new[~bright_mask] = g[~bright_mask]
        r_new[~bright_mask] = r[~bright_mask]
    
    # 防溢出
    b_new = np.clip(b_new, 0, 255).astype(np.uint8)
    g_new = np.clip(g_new, 0, 255).astype(np.uint8)
    r_new = np.clip(r_new, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_new, g_new, r_new])

def perfect_reflector_awb(img, top_percent=0.03):
    """完美反射体白平衡算法（优化默认参数，适配你这张图）"""
    img_float = img.astype(np.float32)
    sum_img = np.sum(img_float, axis=2)
    
    flat_sum = sum_img.flatten()
    threshold = np.percentile(flat_sum, 100 * (1 - top_percent))
    
    mask = sum_img >= threshold
    
    if np.sum(mask) == 0:
        return img
        
    mean_b = np.mean(img_float[:, :, 0][mask])
    mean_g = np.mean(img_float[:, :, 1][mask])
    mean_r = np.mean(img_float[:, :, 2][mask])
    
    max_val = np.max([mean_b, mean_g, mean_r])
    
    img_float[:, :, 0] *= (max_val / mean_b)
    img_float[:, :, 1] *= (max_val / mean_g)
    img_float[:, :, 2] *= (max_val / mean_r)
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

def sobel_edge_detection(img, threshold=80):
    """Sobel边缘检测（提高默认阈值，过滤暖色调纹理边缘）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_abs_x = cv2.convertScaleAbs(grad_x)
    grad_abs_y = cv2.convertScaleAbs(grad_y)
    edge = cv2.addWeighted(grad_abs_x, 0.5, grad_abs_y, 0.5, 0)
    _, edge_mask = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)
    edge_mask = edge_mask > 0
    return edge_mask

def gray_edge_awb(img, edge_threshold=80, use_optimized=False, bright_protect=True):
    """
    灰度边缘白平衡算法（新增亮区保护+阈值优化）
    :param img: 输入BGR图像
    :param edge_threshold: 边缘检测阈值
    :param use_optimized: 是否使用优化版
    :param bright_protect: 是否开启高亮区域保护
    """
    # 提取边缘掩码
    edge_mask = sobel_edge_detection(img, threshold=edge_threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_mask = gray < 200 if bright_protect else np.ones_like(gray, dtype=bool)
    
    # 最终掩码：边缘区域 + 非高亮区域
    final_mask = edge_mask & bright_mask
    
    # 无有效边缘，退化为灰度世界
    if np.sum(final_mask) == 0:
        return gray_world_awb(img, use_optimized=use_optimized, bright_protect=bright_protect)
    
    # 拆分通道，仅计算边缘区域的均值
    b, g, r = cv2.split(img)
    mean_b = np.mean(b[final_mask])
    mean_g = np.mean(g[final_mask])
    mean_r = np.mean(r[final_mask])
    
    # 增益计算
    if use_optimized:
        mean_gray = (mean_b + mean_g + mean_r) / 3.0
    else:
        mean_gray = (mean_b + mean_g + mean_r) / 3.0
    
    kb = mean_gray / (mean_b + 1e-6)
    kg = mean_gray / (mean_g + 1e-6)
    kr = mean_gray / (mean_r + 1e-6)
    
    # 应用增益
    b_new = b.astype(float) * kb
    g_new = g.astype(float) * kg
    r_new = r.astype(float) * kr
    
    # 高亮区域恢复原色
    if bright_protect:
        b_new[~bright_mask] = b[~bright_mask]
        g_new[~bright_mask] = g[~bright_mask]
        r_new[~bright_mask] = r[~bright_mask]
    
    # 防溢出
    b_new = np.clip(b_new, 0, 255).astype(np.uint8)
    g_new = np.clip(g_new, 0, 255).astype(np.uint8)
    r_new = np.clip(r_new, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_new, g_new, r_new])