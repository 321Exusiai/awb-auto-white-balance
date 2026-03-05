import cv2
import numpy as np

def perfect_reflector_awb(image_path, top_percent=0.05):
    """
    完美反射算法自动白平衡
    :param image_path: 图片路径
    :param top_percent: 取亮度前多少的像素作为“完美反射体”，默认0.05（前5%）
    :return: 原图, 校正后图
    """
    # 读取图片（支持中文路径）
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 读取图片出错: {image_path}, 错误: {e}")
        return None, None
    
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None, None
    
    # 分离B、G、R通道
    b, g, r = cv2.split(img)
    
    # 1. 计算灰度图，找到亮度前top_percent的像素
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 把像素展平，排序，取前top_percent的阈值
    flat_gray = gray.ravel()
    threshold = np.percentile(flat_gray, 100 * (1 - top_percent))
    # 生成掩码：只保留亮度>=阈值的像素
    mask = gray >= threshold
    
    # 2. 计算这些像素的RGB平均值
    b_ref = np.mean(b[mask])
    g_ref = np.mean(g[mask])
    r_ref = np.mean(r[mask])
    
    # 3. 计算增益系数（假设参考点应该是白色，即R=G=B=255）
    # 取三个参考值的最大值作为基准，避免过度校正
    max_ref = max(b_ref, g_ref, r_ref)
    gain_b = max_ref / (b_ref + 1e-6)
    gain_g = max_ref / (g_ref + 1e-6)
    gain_r = max_ref / (r_ref + 1e-6)
    
    # 4. 应用增益并防溢出
    b_new = np.clip(b.astype(float) * gain_b, 0, 255)
    g_new = np.clip(g.astype(float) * gain_g, 0, 255)
    r_new = np.clip(r.astype(float) * gain_r, 0, 255)
    
    # 合并通道
    img_awb = cv2.merge([b_new.astype(np.uint8), 
                          g_new.astype(np.uint8), 
                          r_new.astype(np.uint8)])
    return img, img_awb