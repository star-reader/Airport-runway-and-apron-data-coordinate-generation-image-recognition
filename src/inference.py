import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from geojson_generator import GeoJSONGenerator
from model import UNet

def visualize_prediction(image_path, mask, output_path=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = mask.shape
    vis_mask = np.zeros((height, width, 3), dtype=np.uint8)
    vis_mask[mask == 1] = [255, 0, 0]
    vis_mask[mask == 2] = [0, 255, 0]
    vis_mask[mask == 3] = [0, 0, 255]
    image_resized = cv2.resize(image, (width, height))
    alpha = 0.5
    overlay = cv2.addWeighted(image_resized, 1 - alpha, vis_mask, alpha, 0)
    edges = np.zeros_like(vis_mask)
    for i in range(1, 4):
        mask_i = (mask == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours, -1, (255, 255, 255), 2)
    
    # 结合边界和原始图像
    boundary_overlay = image_resized.copy()
    boundary_overlay[edges[:,:,0] > 0] = [255, 255, 255]
    # 创建三张子图
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 3, 1)
    plt.title('原始图像')
    plt.imshow(image_resized)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('分割结果')
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('边界检测')
    plt.imshow(boundary_overlay)
    plt.axis('off')
    plt.tight_layout()

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_visualization.png"
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"可视化结果已保存到 {output_path}")

def main():
    parser = argparse.ArgumentParser(description='推理并生成机场设施的GeoJSON数据')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default='../results', help='输出目录')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generator = GeoJSONGenerator(args.model)
    mask = generator.predict(args.image)
    vis_output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image))[0]}_vis.png")
    visualize_prediction(args.image, mask, vis_output_path)
    json_output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image))[0]}_features.geojson")
    generator.process_image(args.image, json_output_path)

if __name__ == "__main__":
    main()
