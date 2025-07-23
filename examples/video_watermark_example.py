#!/usr/bin/env python3
# coding=utf-8
"""
视频水印示例脚本
演示如何使用blind_watermark库在视频中嵌入和提取水印
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blind_watermark import WaterMark

def main():
    """
    主函数：演示视频水印的嵌入和提取
    """
    print("=== 视频水印示例 ===")
    
    # 创建示例文件
    watermark_path = "wm.png"
    video_path = "input.mp4"
    
    # 输出文件路径
    output_video_path = "output_video_with_watermark.mp4"
    extracted_watermark_path = "extracted_watermark.png"
    
    try:
        # 创建水印对象
        wm = WaterMark(password_wm=12345, password_img=67890)
        
        print("\n1. 嵌入视频水印...")
        # 在视频中嵌入水印
        wm.embed_video(
            wm_path=watermark_path,
            video_path=video_path,
            output_path=output_video_path,
            compression_ratio=95  # 设置压缩质量
        )
        
        # 获取实际水印尺寸
        actual_wm_shape = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE).shape
        print(f"实际水印尺寸: {actual_wm_shape}")

        print("\n2. 提取视频水印...")
        # 从视频中提取水印
        extracted_wm = wm.extract_video(
            video_path=output_video_path,
            wm_shape=actual_wm_shape,  # 水印尺寸
            out_wm_name=extracted_watermark_path,
            mode='img',
            sample_frames=20  # 采样20帧来提取水印
        )
        
        print("\n3. 比较原始水印和提取的水印...")
        # 读取原始水印和提取的水印进行比较
        original_wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        extracted_wm_img = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)
        
        if original_wm is not None and extracted_wm_img is not None:
            # 计算相似度
            similarity = np.corrcoef(original_wm.flatten(), extracted_wm_img.flatten())[0, 1]
            print(f"水印相似度: {similarity:.4f}")
            
            if similarity > 0.7:
                print("✅ 水印提取成功！相似度较高。")
            else:
                print("⚠️  水印提取可能存在问题，相似度较低。")
        else:
            print("❌ 无法读取水印图片进行比较。")
        
        print(f"\n=== 处理完成 ===")
        print(f"原始视频: {video_path}")
        print(f"带水印视频: {output_video_path}")
        print(f"原始水印: {watermark_path}")
        print(f"提取水印: {extracted_watermark_path}")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件（可选）
        # 取消注释以下行来删除临时文件
        # for file_path in [watermark_path, video_path, output_video_path, extracted_watermark_path]:
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        pass


if __name__ == "__main__":
    main() 