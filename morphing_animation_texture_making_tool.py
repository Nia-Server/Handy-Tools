import cv2
import numpy as np
import os

def blend_images(image1_path, image2_path, output_prefix, steps=10):
    # 读取两张带透明度的 PNG 图像
    image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
    
    # 确保尺寸一致
    if image1.shape != image2.shape:
        raise ValueError("两张图像的尺寸不一致，请调整为相同尺寸。")
    
    h, w, _ = image1.shape
    blended_images = []
    
    # 生成混合图片
    for i in range(steps + 1):
        alpha = (steps - i) / steps  # 第一个图的权重
        beta = i / steps  # 第二个图的权重
        
        # 分离 RGB 和 Alpha 通道
        img1_rgb, img1_alpha = image1[:, :, :3], image1[:, :, 3] / 255.0
        img2_rgb, img2_alpha = image2[:, :, :3], image2[:, :, 3] / 255.0
        
        # 混合 RGB
        blended_rgb = cv2.addWeighted(img1_rgb, alpha, img2_rgb, beta, 0)
        
        # 计算 Alpha 通道的混合
        blended_alpha = img1_alpha * alpha + img2_alpha * beta
        blended_alpha = (blended_alpha * 255).astype(np.uint8)
        
        # 组合最终的 RGBA 图像
        blended_image = np.dstack((blended_rgb, blended_alpha))
        blended_images.append(blended_image)
        
        output_path = f"{output_prefix}_{int(alpha*100)}_{int(beta*100)}.png"
        cv2.imwrite(output_path, blended_image)
        print(f"Saved: {output_path}")
    
    return blended_images

def merge_images(images, output_path):
    """按顺序将图像垂直拼接"""
    merged_image = np.vstack(images)
    cv2.imwrite(output_path, merged_image)
    print(f"Merged image saved: {output_path}")

# 示例调用
blended_images = blend_images("image1.png", "image2.png", "output/blended", steps=10)
merge_images(blended_images, "output/merged_forward.png")
merge_images(blended_images[::-1], "output/merged_reverse.png")
