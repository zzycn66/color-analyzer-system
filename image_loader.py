"""
image_loader.py
功能：负责图像的读取、预处理和基础变换
"""

import os
import cv2
import numpy as np


class ImageLoader:
    """图像加载器，负责读取和预处理图像"""

    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.file_path = None
        self.file_name = None

    def load_image(self, file_path):
        """
        从指定路径加载图像
        :param file_path: 图像文件路径
        :return: 加载是否成功
        """
        if not os.path.exists(file_path):
            print(f"[错误] 文件不存在: {file_path}")
            return False

        _, ext = os.path.splitext(file_path)
        if ext.lower() not in self.SUPPORTED_FORMATS:
            print(f"[错误] 不支持的格式: {ext}")
            print(f"       支持的格式: {', '.join(self.SUPPORTED_FORMATS)}")
            return False

        # 用 numpy 绕过中文路径问题
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        self.original_image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        if self.original_image is None:
            print(f"[错误] 图像读取失败: {file_path}")
            return False

        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.processed_image = self.original_image.copy()

        h, w = self.original_image.shape[:2]
        print(f"[信息] 成功加载图像: {self.file_name}")
        print(f"       尺寸: {w} x {h}")
        return True

    def resize(self, target_width=640):
        """
        等比例缩放图像到指定宽度
        :param target_width: 目标宽度，默认640像素
        """
        if self.processed_image is None:
            print("[错误] 未加载图像，无法缩放")
            return

        h, w = self.processed_image.shape[:2]
        ratio = target_width / w
        target_height = int(h * ratio)

        self.processed_image = cv2.resize(
            self.processed_image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        print(f"[信息] 图像已缩放至: {target_width} x {target_height}")

    def apply_blur(self, kernel_size=5):
        """
        对图像应用高斯模糊以降噪
        :param kernel_size: 模糊核大小，必须为奇数
        """
        if self.processed_image is None:
            print("[错误] 未加载图像，无法模糊")
            return

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.processed_image = cv2.GaussianBlur(
            self.processed_image,
            (kernel_size, kernel_size),
            0
        )
        print(f"[信息] 已应用高斯模糊，核大小: {kernel_size}")

    def convert_color_space(self, target_space='rgb'):
        """
        转换图像色彩空间
        :param target_space: 目标色彩空间 ('rgb', 'hsv', 'lab', 'gray')
        """
        if self.processed_image is None:
            print("[错误] 未加载图像")
            return

        space_map = {
            'rgb': cv2.COLOR_BGR2RGB,
            'hsv': cv2.COLOR_BGR2HSV,
            'lab': cv2.COLOR_BGR2LAB,
            'gray': cv2.COLOR_BGR2GRAY,
        }

        code = space_map.get(target_space.lower())
        if code is None:
            print(f"[错误] 不支持的色彩空间: {target_space}")
            return

        self.processed_image = cv2.cvtColor(self.processed_image, code)
        print(f"[信息] 色彩空间已转换为: {target_space.upper()}")

    def get_image_data(self):
        """获取当前处理后的图像数据"""
        return self.processed_image

    def get_original_data(self):
        """获取原始图像数据"""
        return self.original_image

    def get_image_info(self):
        """返回图像的基本信息字典"""
        if self.processed_image is None:
            return None

        img = self.processed_image
        info = {
            'file_name': self.file_name,
            'height': img.shape[0],
            'width': img.shape[1],
            'channels': img.shape[2] if len(img.shape) == 3 else 1,
            'total_pixels': img.shape[0] * img.shape[1],
            'dtype': str(img.dtype),
        }
        return info

    def reset(self):
        """重置为原始图像"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            print("[信息] 已重置为原始图像")
