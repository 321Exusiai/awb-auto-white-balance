import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # 新增：适配文件夹路径

# ========== 解决matplotlib中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ========== 适配你的文件夹结构（固定相对路径）==========
# 原始图片文件夹
ORIGINAL_FOLDER = "original"
# 校正后图片保存文件夹
OPTIMIZED_FOLDER = "optimized"
# 对比结果图保存文件夹
CONTRAST_FOLDER = "contrast"

# 自动创建文件夹（不存在就新建，避免报错）
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(OPTIMIZED_FOLDER, exist_ok=True)
os.makedirs(CONTRAST_FOLDER, exist_ok=True)
# =================================================

# ===================== 1. 核心算法 =====================
def gray_world_awb(image_path):
    # 使用 numpy 读取图片以支持中文路径
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取图片时发生错误：{e}")
        return None, None
    if img is None:
        print(f"错误：无法读取图片 {image_path}，请检查路径及文件完整性！")
        return None, None  # 改为返回两个None，避免解包错误
    # 分离B、G、R通道
    b, g, r = cv2.split(img)
    # 计算各通道平均值（除以M×N，对应全彩图像的总像素数）
    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)
    # 计算全局中性灰基准K
    K = (b_avg + g_avg + r_avg) / 3.0
    # 计算增益系数（+1e-6防除零）
    gain_b = K / (b_avg + 1e-6)
    gain_g = K / (g_avg + 1e-6)
    gain_r = K / (r_avg + 1e-6)
    # 应用增益系数（先转float避免精度丢失）
    b_new = b.astype(float) * gain_b
    g_new = g.astype(float) * gain_g
    r_new = r.astype(float) * gain_r
    # 像素值防溢出（截断到0-255）
    b_new = np.clip(b_new, 0, 255)
    g_new = np.clip(g_new, 0, 255)
    r_new = np.clip(r_new, 0, 255)
    # 合并通道，转回uint8格式（OpenCV要求的图像格式）
    img_awb = cv2.merge([b_new.astype(np.uint8), g_new.astype(np.uint8), r_new.astype(np.uint8)])
    return img, img_awb  # 返回原图和校正图

# ===================== 2. 可视化对比 =====================
def show_comparison(original_img, awb_img, img_name):
    # 转换颜色空间（OpenCV BGR → matplotlib RGB）
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    awb_rgb = cv2.cvtColor(awb_img, cv2.COLOR_BGR2RGB)
    # 转换为灰度图（用于直方图）
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    awb_gray = cv2.cvtColor(awb_img, cv2.COLOR_BGR2GRAY)
    # 创建画布，显示4个内容：原图、校正图、原图直方图、校正图直方图
    plt.figure(figsize=(16, 8))
    # 1. 显示原图
    plt.subplot(2, 2, 1)
    plt.imshow(original_rgb)
    plt.title(f'原图：{img_name}')
    plt.axis('off')
    # 2. 显示校正图
    plt.subplot(2, 2, 2)
    plt.imshow(awb_rgb)
    plt.title(f'白平衡校正后')
    plt.axis('off')
    # 3. 原图亮度直方图
    plt.subplot(2, 2, 3)
    plt.hist(original_gray.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.title('原图亮度直方图')
    plt.xlabel('亮度值')
    plt.ylabel('像素数量')
    # 4. 校正图亮度直方图
    plt.subplot(2, 2, 4)
    plt.hist(awb_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title('校正后亮度直方图')
    plt.xlabel('亮度值')
    plt.ylabel('像素数量')
    # 保存对比图到contrast文件夹
    plt.tight_layout()
    save_contrast_path = os.path.join(CONTRAST_FOLDER, f'{img_name}_对比结果.png')
    plt.savefig(save_contrast_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 对比图已保存：{save_contrast_path}")

# ===================== 3. 主函数（运行测试） =====================
if __name__ == "__main__":
    # ------------ 适配你的文件夹，不用改路径！------------
    # 格式：(original里的图片文件名, 图片显示名称)
    test_images = [
        ("01_过暖.JPG", "暖光偏黄图"),
        ("02_鲜艳.JPG", "颜色丰富图"),
        ("03_单色.JPG", "颜色单一图_1"),
        ("04_双色.JPG", "颜色单一图_2"),
        ("05_冷色.JPG", "冷光偏蓝图"),
    ]
    
    # 循环处理每张图（增加错误判断）
    for img_filename, img_name in test_images:
        # 拼接原始图片完整路径
        img_path = os.path.join(ORIGINAL_FOLDER, img_filename)
        original, awb_result = gray_world_awb(img_path)
        # 先判断是否读取成功
        if original is None or awb_result is None:
            print(f"❌ 处理失败：{img_name}（图片读取错误）\n")
            continue  # 跳过当前图片，处理下一张
        # 读取成功则执行后续逻辑
        print(f"✅ 成功处理：{img_name}")
        show_comparison(original, awb_result, img_name)
        # 保存校正后的图片到optimized文件夹（支持中文路径）
        save_img_path = os.path.join(OPTIMIZED_FOLDER, f'{img_name}_校正后.jpg')
        _, img_encode = cv2.imencode('.jpg', awb_result)
        img_encode.tofile(save_img_path)
        print(f"✅ 校正图已保存：{save_img_path}\n")