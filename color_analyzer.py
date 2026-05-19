"""
color_analyzer.py
功能：使用K-Means聚类提取图像中的主色，并计算各颜色的占比
"""

import numpy as np
from collections import Counter


class ColorAnalyzer:
    """颜色分析器，通过K-Means聚类识别图像中的主要颜色块"""

    # 预定义的常见颜色名称及BGR参考值
    COLOR_NAMES_CN = {
        '红色': (0, 0, 255),
        '橙色': (0, 128, 255),
        '黄色': (0, 255, 255),
        '绿色': (0, 200, 0),
        '青色': (255, 200, 0),
        '蓝色': (255, 0, 0),
        '紫色': (255, 0, 128),
        '白色': (255, 255, 255),
        '灰色': (128, 128, 128),
        '黑色': (0, 0, 0),
        '粉色': (203, 192, 255),
        '棕色': (42, 42, 165),
    }

    def __init__(self, image_data):
        """
        初始化分析器
        :param image_data: BGR格式的numpy图像数组
        """
        self.image_data = image_data
        self.labels = None
        self.centers = None
        self.num_clusters = 0
        self.color_proportions = None

    def _preprocess_pixels(self):
        """将图像数据转为二维像素矩阵，每行为一个BGR像素点"""
        pixels = self.image_data.reshape(-1, 3).astype(np.float32)
        return pixels

    def analyze(self, num_colors=5, max_iterations=100):
        """
        执行K-Means颜色聚类分析
        :param num_colors: 要提取的颜色数量
        :param max_iterations: 最大迭代次数
        :return: 颜色分析结果列表
        """
        if self.image_data is None:
            print("[错误] 没有可用的图像数据")
            return None

        self.num_clusters = num_colors
        pixels = self._preprocess_pixels()
        total_pixels = len(pixels)

        print(f"[信息] 开始颜色聚类分析，提取 {num_colors} 种主要颜色...")
        print(f"       总像素数: {total_pixels}")

        # 手动实现K-Means聚类
        # 随机选择初始聚类中心
        np.random.seed(42)
        random_indices = np.random.choice(total_pixels, num_colors, replace=False)
        self.centers = pixels[random_indices].copy()

        for iteration in range(max_iterations):
            # 计算每个像素到各聚类中心的距离
            distances = np.zeros((total_pixels, num_colors))
            for i in range(num_colors):
                diff = pixels - self.centers[i]
                distances[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))

            # 分配像素到最近的聚类中心
            self.labels = np.argmin(distances, axis=1)

            # 更新聚类中心
            new_centers = np.zeros_like(self.centers)
            for i in range(num_colors):
                cluster_pixels = pixels[self.labels == i]
                if len(cluster_pixels) > 0:
                    new_centers[i] = np.mean(cluster_pixels, axis=0)
                else:
                    new_centers[i] = self.centers[i]

            # 检查是否收敛
            shift = np.max(np.sqrt(np.sum((new_centers - self.centers) ** 2, axis=1)))
            self.centers = new_centers

            if iteration % 10 == 0:
                print(f"       迭代 {iteration}: 最大中心偏移 = {shift:.4f}")

            if shift < 0.5:
                print(f"       已在第 {iteration} 次迭代收敛")
                break

        # 计算各颜色的占比
        self._calculate_proportions(total_pixels)
        print("[信息] 颜色分析完成")
        return self.color_proportions

    def _calculate_proportions(self, total_pixels):
        """计算并存储各颜色的占比信息"""
        label_counts = Counter(self.labels)
        self.color_proportions = []

        for cluster_id in range(self.num_clusters):
            count = label_counts.get(cluster_id, 0)
            ratio = count / total_pixels
            bgr_color = self.centers[cluster_id].astype(int)
            color_name = self._identify_color_name(bgr_color)

            self.color_proportions.append({
                'cluster_id': cluster_id,
                'bgr': tuple(bgr_color.tolist()),
                'rgb': (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])),
                'hex': '#{:02x}{:02x}{:02x}'.format(
                    int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])
                ),
                'name': color_name,
                'pixel_count': count,
                'proportion': round(ratio, 6),
                'percentage': round(ratio * 100, 2),
            })

        # 按占比降序排列
        self.color_proportions.sort(key=lambda x: x['proportion'], reverse=True)

    def _identify_color_name(self, bgr):
        """
        根据BGR值匹配最接近的预定义颜色名称
        :param bgr: BGR颜色值 (B, G, R)
        :return: 颜色名称字符串
        """
        min_dist = float('inf')
        matched_name = '未知'

        for name, ref_bgr in self.COLOR_NAMES_CN.items():
            dist = sum((int(bgr[i]) - int(ref_bgr[i])) ** 2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                matched_name = name

        return matched_name

    def get_sorted_colors(self):
        """获取按占比排序的颜色列表"""
        if self.color_proportions is None:
            print("[警告] 尚未执行分析，请先调用 analyze()")
            return []
        return self.color_proportions

    def get_label_image(self):
        """返回聚类标签图，每个像素被替换为其所属聚类中心颜色"""
        if self.labels is None or self.centers is None:
            return None

        result = np.zeros_like(self.image_data.reshape(-1, 3))
        for i in range(self.num_clusters):
            result[self.labels == i] = self.centers[i].astype(np.uint8)

        result = result.reshape(self.image_data.shape)
        return result

    def print_report(self):
        """在终端打印颜色分析报告"""
        if self.color_proportions is None:
            print("[警告] 尚未执行分析")
            return

        print("\n" + "=" * 60)
        print("          色块占比分析报告")
        print("=" * 60)
        print(f"{'序号':<5}{'颜色':<8}{'HEX':<10}{'RGB':<20}{'占比':<10}{'像素数'}")
        print("-" * 60)

        for i, item in enumerate(self.color_proportions):
            rgb_str = f"({item['rgb'][0]}, {item['rgb'][1]}, {item['rgb'][2]})"
            print(f"{i + 1:<5}"
                  f"{item['name']:<8}"
                  f"{item['hex']:<10}"
                  f"{rgb_str:<20}"
                  f"{item['percentage']:<10}%"
                  f"{item['pixel_count']}")

        print("=" * 60 + "\n")
