#!/usr/bin/env python3
# coding=utf-8
# @Time    : 2020/8/13
# @Author  : github.com/guofei9987
import warnings

import numpy as np
import cv2

from .bwm_core import WaterMarkCore
from .version import bw_notes

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，创建一个简单的进度条替代
    class tqdm:
        def __init__(self, total=None, **kwargs):
            self.total = total
            self.count = 0
        
        def update(self, n=1):
            self.count += n
            if self.total:
                print(f"\r进度: {self.count}/{self.total} ({self.count/self.total*100:.1f}%)", end="")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            print()


class WaterMark:
    def __init__(self, password_wm=1, password_img=1, block_shape=(4, 4), mode='common', processes=None):
        bw_notes.print_notes()

        self.bwm_core = WaterMarkCore(password_img=password_img, mode=mode, processes=processes)

        self.password_wm = password_wm

        self.wm_bit = None
        self.wm_size = 0

    def read_img(self, filename=None, img=None):
        if img is None:
            # 从文件读入图片
            img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            assert img is not None, "image file '{filename}' not read".format(filename=filename)

        self.bwm_core.read_img_arr(img=img)
        return img

    def read_wm(self, wm_content, mode='img'):
        assert mode in ('img', 'str', 'bit'), "mode in ('img','str','bit')"
        if mode == 'img':
            wm = cv2.imread(filename=wm_content, flags=cv2.IMREAD_GRAYSCALE)
            assert wm is not None, 'file "{filename}" not read'.format(filename=wm_content)

            # 读入图片格式的水印，并转为一维 bit 格式，抛弃灰度级别
            self.wm_bit = wm.flatten() > 128

        elif mode == 'str':
            byte = bin(int(wm_content.encode('utf-8').hex(), base=16))[2:]
            self.wm_bit = (np.array(list(byte)) == '1')
        else:
            self.wm_bit = np.array(wm_content)

        self.wm_size = self.wm_bit.size

        # 水印加密:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

        self.bwm_core.read_wm(self.wm_bit)

    def embed(self, filename=None, compression_ratio=None):
        '''
        :param filename: string
            Save the image file as filename
        :param compression_ratio: int or None
            If compression_ratio = None, do not compression,
            If compression_ratio is integer between 0 and 100, the smaller, the output file is smaller.
        :return:
        '''
        embed_img = self.bwm_core.embed()
        if filename is not None:
            if compression_ratio is None:
                cv2.imwrite(filename=filename, img=embed_img)
            elif filename.endswith('.jpg'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_JPEG_QUALITY, compression_ratio])
            elif filename.endswith('.png'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_PNG_COMPRESSION, compression_ratio])
            else:
                cv2.imwrite(filename=filename, img=embed_img)
        return embed_img

    def embed_video(self, wm_path, video_path, output_path, compression_ratio=None, preserve_codec=True):
        """
        在视频中嵌入水印
        
        :param wm_path: str
            水印图片路径
        :param video_path: str
            输入视频路径
        :param output_path: str
            输出视频路径
        :param compression_ratio: int or None
            压缩比例，None表示不压缩
        :param preserve_codec: bool
            是否保持原始编码格式，默认为True
        :return: None
        """
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 读取水印
        self.read_wm(wm_path, mode='img')
        
        # 创建视频写入器
        # 保持原始编码格式，避免编码改变
        if preserve_codec:
            # 保持原始编码格式
            if output_path.endswith('.mp4'):
                # 检查原始编码，如果是H.264则保持H.264
                fourcc_str = cv2.VideoWriter_fourcc(*'H264')
                # 尝试使用H.264编码器
                try:
                    out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
                    if not out.isOpened():
                        # 如果H.264不可用，尝试avc1
                        fourcc_str = cv2.VideoWriter_fourcc(*'avc1')
                        out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
                        if not out.isOpened():
                            # 如果都不可用，使用mp4v
                            fourcc_str = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
                except:
                    # 如果出现异常，使用mp4v
                    fourcc_str = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
            elif output_path.endswith('.avi'):
                fourcc_str = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
            else:
                fourcc_str = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
        else:
            # 不保持原始编码，使用默认设置
            if output_path.endswith('.mp4'):
                fourcc_str = cv2.VideoWriter_fourcc(*'mp4v')
            elif output_path.endswith('.avi'):
                fourcc_str = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc_str = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc_str, fps, frame_size)
        if not out.isOpened():
            raise ValueError(f"无法创建输出视频文件: {output_path}")
        
        # 处理每一帧
        count = 0
        with tqdm(total=total_frames, desc="嵌入视频水印") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    count += 1
                    
                    # 将帧转换为float32类型进行处理
                    frame_float = frame.astype(np.float32)
                    
                    # 读取当前帧到水印核心
                    self.bwm_core.read_img_arr(img=frame_float)
                    
                    # 嵌入水印
                    wmed_frame = self.bwm_core.embed()
                    
                    # 确保像素值在有效范围内
                    wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
                    wmed_frame = np.around(wmed_frame).astype(np.uint8)
                    
                    # 写入输出视频
                    out.write(wmed_frame)
                    
                    pbar.update(1)
                    
                    # 检查是否按下q键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
        
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"视频水印嵌入完成！输出文件: {output_path}")

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode='img'):
        assert wm_shape is not None, 'wm_shape needed'

        if filename is not None:
            embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
            assert embed_img is not None, "{filename} not read".format(filename=filename)

        self.wm_size = np.array(wm_shape).prod()

        if mode in ('str', 'bit'):
            wm_avg = self.bwm_core.extract_with_kmeans(img=embed_img, wm_shape=wm_shape)
        else:
            wm_avg = self.bwm_core.extract(img=embed_img, wm_shape=wm_shape)

        # 解密：
        wm = self.extract_decrypt(wm_avg=wm_avg)

        # 转化为指定格式：
        if mode == 'img':
            wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
            cv2.imwrite(out_wm_name, wm)
        elif mode == 'str':
            byte = ''.join(str((i >= 0.5) * 1) for i in wm)
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode('utf-8', errors='replace')

        return wm

    def extract_video(self, video_path, wm_shape, out_wm_name=None, mode='img', sample_frames=10):
        """
        从视频中提取水印
        
        :param video_path: str
            输入视频路径
        :param wm_shape: tuple
            水印形状 (height, width)
        :param out_wm_name: str
            输出水印文件名
        :param mode: str
            输出模式 ('img', 'str', 'bit')
        :param sample_frames: int
            采样帧数，用于提取水印
        :return: 提取的水印
        """
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        if sample_frames >= total_frames:
            sample_frames = total_frames
            frame_interval = 1
        else:
            frame_interval = total_frames // sample_frames
        
        # 存储所有提取的水印
        extracted_wms = []
        
        with tqdm(total=sample_frames, desc="提取视频水印") as pbar:
            frame_count = 0
            sample_count = 0
            
            while cap.isOpened() and sample_count < sample_frames:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    
                    # 按间隔采样帧
                    if frame_count % frame_interval == 0:
                        # 提取当前帧的水印
                        wm_avg = self.bwm_core.extract(img=frame, wm_shape=wm_shape)
                        extracted_wms.append(wm_avg)
                        sample_count += 1
                        pbar.update(1)
                else:
                    break
        
        cap.release()
        
        if not extracted_wms:
            raise ValueError("未能从视频中提取到水印")
        
        # 对所有提取的水印求平均
        avg_wm = np.mean(extracted_wms, axis=0)
        
        # 解密水印
        wm = self.extract_decrypt(wm_avg=avg_wm)
        
        # 转化为指定格式
        if mode == 'img':
            wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
            if out_wm_name:
                cv2.imwrite(out_wm_name, wm)
        elif mode == 'str':
            byte = ''.join(str((i >= 0.5) * 1) for i in wm)
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode('utf-8', errors='replace')
        
        print(f"视频水印提取完成！")
        return wm
