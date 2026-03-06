import cv2
import numpy as np

def gray_world_awb(img, use_optimized=False):
    """灰度世界白平衡算法"""
    b, g, r = cv2.split(img)
    if use_optimized:
        # 优化版：分通道计算平均值
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        mean_gray = (mean_b + mean_g + mean_r) / 3
        
        b = cv2.addWeighted(b, mean_gray / mean_b, 0, 0, 0)
        g = cv2.addWeighted(g, mean_gray / mean_g, 0, 0, 0)
        r = cv2.addWeighted(r, mean_gray / mean_r, 0, 0, 0)
    else:
        # 基础版
        mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
        kb, kg, kr = (mean_b + mean_g + mean_r) / (3 * mean_b), (mean_b + mean_g + mean_r) / (3 * mean_g), (mean_b + mean_g + mean_r) / (3 * mean_r)
        b, g, r = np.uint8(b * kb), np.uint8(g * kg), np.uint8(r * kr)
    
    return cv2.merge([b, g, r])

def perfect_reflector_awb(img, top_percent=0.05):
    """完美反射体控制白平衡"""
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
