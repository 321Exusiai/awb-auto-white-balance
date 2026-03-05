import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ========== 解决matplotlib中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 适配你的文件夹结构（固定相对路径）==========
ORIGINAL_FOLDER = "original"
OPTIMIZED_FOLDER = "optimized"
CONTRAST_FOLDER = "contrast"

os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(OPTIMIZED_FOLDER, exist_ok=True)
os.makedirs(CONTRAST_FOLDER, exist_ok=True)
# =================================================

# ===================== 1. 核心算法（支持3种模式）=====================
def awb_algorithm(image_path, use_optimized=False, use_white_patch=False):
    """
    白平衡核心算法：支持灰度世界（原版/优化版）、完美反射体（取最亮值）
    :param image_path: 图片路径
    :param use_optimized: 灰度世界是否用优化版（中亮度区域）
    :param use_white_patch: 是否使用完美反射体算法（取最亮值）
    :return: 原图, 校正后图
    """
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 读取图片出错: {image_path}, 错误: {e}")
        return None, None
    
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None, None
    
    b, g, r = cv2.split(img)
    
    # -------- 模式1：完美反射体（取最亮的点） --------
    if use_white_patch:
        # 取每个通道的最大值（最亮值），也可改为取前N%亮的像素均值（更鲁棒）
        # 鲁棒版：取亮度前1%的像素均值（避免单像素噪声）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        top_percent = 1  # 取最亮的1%像素
        threshold = np.percentile(gray, 100 - top_percent)
        mask = gray >= threshold
        
        if np.sum(mask) == 0:  # 无满足条件的像素，退化为全局最大值
            b_max = np.max(b)
            g_max = np.max(g)
            r_max = np.max(r)
            print(f"   ✨ 使用完美反射体（全局最亮值）")
        else:
            b_max = np.mean(b[mask])
            g_max = np.mean(g[mask])
            r_max = np.mean(r[mask])
            print(f"   ✨ 使用完美反射体（最亮{top_percent}%像素均值）")
        
        # 完美反射体增益计算：目标是让最亮值趋近255
        gain_b = 255.0 / (b_max + 1e-6)
        gain_g = 255.0 / (g_max + 1e-6)
        gain_r = 255.0 / (r_max + 1e-6)
    
    # -------- 模式2：灰度世界（原版/优化版） --------
    else:
        if use_optimized:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = (gray >= 60) & (gray <= 180)
            valid_ratio = np.sum(mask)/mask.size
            if valid_ratio >= 0.1:
                b_avg = np.mean(b[mask])
                g_avg = np.mean(g[mask])
                r_avg = np.mean(r[mask])
                print(f"   ✨ 使用灰度世界优化版（中亮度像素占比: {valid_ratio:.1%}）")
            else:
                b_avg = np.mean(b)
                g_avg = np.mean(g)
                r_avg = np.mean(r)
                print(f"   ⚠️ 中亮度像素占比不足{valid_ratio:.1%}，切换为灰度世界原版")
        else:
            b_avg = np.mean(b)
            g_avg = np.mean(g)
            r_avg = np.mean(r)
            print(f"   📌 使用灰度世界原版（全局像素均值）")
        
        # 灰度世界增益计算：目标是让RGB均值相等（趋近灰度）
        # 可选：按人眼敏感度加权（G:0.587, R:0.299, B:0.114）
        # K = (0.114*b_avg + 0.587*g_avg + 0.299*r_avg)  # 加权版
        K = (b_avg + g_avg + r_avg) / 3.0  # 等权原版
        gain_b = K / (b_avg + 1e-6)
        gain_g = K / (g_avg + 1e-6)
        gain_r = K / (r_avg + 1e-6)
    
    # -------- 应用增益并裁剪（避免溢出0-255） --------
    b_new = np.clip(b.astype(float) * gain_b, 0, 255)
    g_new = np.clip(g.astype(float) * gain_g, 0, 255)
    r_new = np.clip(r.astype(float) * gain_r, 0, 255)
    
    img_awb = cv2.merge([b_new.astype(np.uint8), 
                          g_new.astype(np.uint8), 
                          r_new.astype(np.uint8)])
    return img, img_awb

# ===================== 2. 可视化对比（兼容新算法）=====================
def show_comparison(original_img, awb_img, img_name, version_suffix):
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    awb_rgb = cv2.cvtColor(awb_img, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    awb_gray = cv2.cvtColor(awb_img, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_rgb)
    plt.title(f'原图: {img_name}')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(awb_rgb)
    plt.title(f'白平衡校正后 ({version_suffix})')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.hist(original_gray.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.title('原图亮度直方图')
    plt.xlabel('亮度值')
    plt.ylabel('像素数量')
    
    plt.subplot(2, 2, 4)
    plt.hist(awb_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title(f'校正后亮度直方图 ({version_suffix})')
    plt.xlabel('亮度值')
    plt.ylabel('像素数量')
    
    save_path = os.path.join(CONTRAST_FOLDER, f'{img_name}_对比结果_{version_suffix}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 对比图已保存: {save_path}")

# ===================== 3. 批量处理（支持多算法）=====================
def process_algorithm(use_optimized=False, use_white_patch=False, version_suffix=""):
    """
    批量处理指定算法
    :param use_optimized: 灰度世界是否优化
    :param use_white_patch: 是否用完美反射体（最亮值）
    :param version_suffix: 版本后缀（区分不同算法）
    """
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    print(f"\n{'='*60}")
    print(f"🚀 开始处理【{version_suffix}】版本...")
    print(f"{'='*60}")
    
    img_count = 0
    for filename in os.listdir(ORIGINAL_FOLDER):
        if filename.endswith(supported_ext):
            img_count += 1
            img_path = os.path.join(ORIGINAL_FOLDER, filename)
            img_name = os.path.splitext(filename)[0]
            
            print(f"\n[{img_count}] 处理图片: {filename}")
            
            original, awb_result = awb_algorithm(
                img_path, 
                use_optimized=use_optimized, 
                use_white_patch=use_white_patch
            )
            if original is None or awb_result is None:
                continue
            
            # 生成对比图
            show_comparison(original, awb_result, img_name, version_suffix)
            
            # 保存校正图
            save_img_path = os.path.join(OPTIMIZED_FOLDER, f'{img_name}_校正后_{version_suffix}.jpg')
            _, img_encode = cv2.imencode('.jpg', awb_result)
            img_encode.tofile(save_img_path)
            print(f"   💾 校正图已保存: {save_img_path}")
    
    print(f"\n{'='*60}")
    if img_count == 0:
        print(f"❌ 【{version_suffix}】版本处理失败：original文件夹里没有找到图片！")
    else:
        print(f"✅ 【{version_suffix}】版本处理完成！共处理 {img_count} 张图片")
    print(f"{'='*60}\n")

# ===================== 4. 主函数（选择运行模式）=====================
if __name__ == "__main__":
    # ============== 【关键设置】选择要运行的算法 ==============
    RUN_GRAY_WORLD_V0 = True    # 灰度世界原版（等权均值）
    RUN_GRAY_WORLD_V2 = True    # 灰度世界优化版（中亮度区域）
    RUN_WHITE_PATCH = True      # 完美反射体（取最亮值）
    RUN_GRAY_WORLD_WEIGHTED = True  # 灰度世界（人眼加权均值）
    # =========================================================
    
    print("\n" + "="*60)
    print("📋 运行配置：")
    if RUN_GRAY_WORLD_V0:
        print("   ✅ 运行【灰度世界原版】（v0）")
    if RUN_GRAY_WORLD_V2:
        print("   ✅ 运行【灰度世界优化版】（v2）")
    if RUN_WHITE_PATCH:
        print("   ✅ 运行【完美反射体（最亮值）】（wp）")
    if RUN_GRAY_WORLD_WEIGHTED:
        print("   ✅ 运行【灰度世界（人眼加权）】（weighted）")
    print("="*60)
    
    # 运行各算法
    if RUN_GRAY_WORLD_V0:
        process_algorithm(use_optimized=False, use_white_patch=False, version_suffix="v0")
    
    if RUN_GRAY_WORLD_V2:
        process_algorithm(use_optimized=True, use_white_patch=False, version_suffix="v2")
    
    if RUN_WHITE_PATCH:
        process_algorithm(use_optimized=False, use_white_patch=True, version_suffix="wp")
    
    if RUN_GRAY_WORLD_WEIGHTED:
        # 先临时修改灰度世界的K为加权版（也可封装为参数）
        original_awb_algorithm = awb_algorithm
        def weighted_awb_algorithm(*args, **kwargs):
            img = cv2.imdecode(np.fromfile(args[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            b, g, r = cv2.split(img)
            b_avg = np.mean(b)
            g_avg = np.mean(g)
            r_avg = np.mean(r)
            # 人眼敏感度加权：G(0.587) > R(0.299) > B(0.114)
            K = 0.114*b_avg + 0.587*g_avg + 0.299*r_avg
            gain_b = K / (b_avg + 1e-6)
            gain_g = K / (g_avg + 1e-6)
            gain_r = K / (r_avg + 1e-6)
            b_new = np.clip(b.astype(float)*gain_b, 0, 255)
            g_new = np.clip(g.astype(float)*gain_g, 0, 255)
            r_new = np.clip(r.astype(float)*gain_r, 0, 255)
            img_awb = cv2.merge([b_new.astype(np.uint8), g_new.astype(np.uint8), r_new.astype(np.uint8)])
            return img, img_awb
        awb_algorithm = weighted_awb_algorithm
        process_algorithm(use_optimized=False, use_white_patch=False, version_suffix="weighted")
        # 恢复原函数
        awb_algorithm = original_awb_algorithm
    
    print("\n🎉 所有任务已完成！")
    print(f"   校正后图片在：{os.path.abspath(OPTIMIZED_FOLDER)}")
    print(f"   对比结果在：{os.path.abspath(CONTRAST_FOLDER)}")