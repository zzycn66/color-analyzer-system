"""
result_visualizer.py
功能：将颜色分析结果进行可视化展示，包括饼图、色卡和HTML报告
"""

import os
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont


class ResultVisualizer:
    """结果可视化器，负责生成分析图表和报告"""

    def __init__(self, image_info, color_results, original_img=None, clustered_img=None, output_dir='output'):
        """
        初始化可视化器
        :param image_info: 图像信息字典
        :param color_results: 颜色分析结果列表
        :param original_img: 原始图像
        :param clustered_img: 聚类后的图像
        :param output_dir: 输出目录
        """
        self.image_info = image_info
        self.color_results = color_results
        self.original_img = original_img
        self.clustered_img = clustered_img
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载中文字体
        self.font = self._load_chinese_font()

    def _load_chinese_font(self, size=16):
        """
        尝试加载系统中文字体，依次尝试多个常见路径
        :param size: 字体大小
        :return: PIL ImageFont 对象
        """
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",      # 微软雅黑粗体
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/STKAITI.TTF",     # 华文楷体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
            "/System/Library/Fonts/PingFang.ttc",             # macOS
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        print("[警告] 未找到中文字体，中文可能无法正常显示")
        return ImageFont.load_default()

    def _cv2_to_pil(self, cv_img):
        """OpenCV图像转PIL图像"""
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def _pil_to_cv2(self, pil_img):
        """PIL图像转OpenCV图像"""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_chinese_text(self, img, text, position, font_size=16, color=(255, 255, 255)):
        """
        在OpenCV图像上绘制中文文字
        :param img: OpenCV BGR图像
        :param text: 文字内容
        :param position: (x, y) 位置
        :param font_size: 字体大小
        :param color: RGB颜色元组
        :return: 处理后的OpenCV图像
        """
        pil_img = self._cv2_to_pil(img)
        draw = ImageDraw.Draw(pil_img)
        font = self._load_chinese_font(font_size)
        draw.text(position, text, font=font, fill=color)
        return self._pil_to_cv2(pil_img)

    def _img_to_base64(self, img):
        """将OpenCV图像转为base64字符串"""
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_color_bar(self, width=600, bar_height=80):
        """
        生成颜色占比条形图
        :param width: 图像宽度
        :param bar_height: 色条高度
        :return: 生成的条形图图像
        """
        num_colors = len(self.color_results)
        header_height = 50
        info_height = 30
        total_height = header_height + bar_height + info_height * num_colors + 40

        canvas = np.ones((total_height, width, 3), dtype=np.uint8) * 45

        # 绘制标题（中文）
        canvas = self._draw_chinese_text(canvas, "色块占比分析", (15, 12), font_size=22)

        # 绘制色条
        x_offset = 20
        bar_y = header_height
        bar_width = width - 40

        for item in self.color_results:
            segment_width = max(int(bar_width * item['proportion']), 1)
            color_bgr = item['bgr']
            cv2.rectangle(canvas,
                          (x_offset, bar_y),
                          (x_offset + segment_width, bar_y + bar_height),
                          color_bgr, -1)

            # 在色块上写百分比
            if segment_width > 50:
                text = f"{item['percentage']}%"
                text_x = x_offset + 5
                text_y = bar_y + bar_height // 2 - 8
                canvas = self._draw_chinese_text(canvas, text, (text_x, text_y),
                                                  font_size=14, color=(255, 255, 255))

            x_offset += segment_width

        # 绘制色条边框
        cv2.rectangle(canvas,
                      (20, header_height),
                      (width - 20, header_height + bar_height),
                      (200, 200, 200), 1)

        # 绘制颜色详情列表
        y_offset = header_height + bar_height + 20
        for i, item in enumerate(self.color_results):
            # 色块小方块
            square_size = 18
            cv2.rectangle(canvas,
                          (20, y_offset),
                          (20 + square_size, y_offset + square_size),
                          item['bgr'], -1)
            cv2.rectangle(canvas,
                          (20, y_offset),
                          (20 + square_size, y_offset + square_size),
                          (180, 180, 180), 1)

            # 文字信息（中文）
            info_text = f"#{i + 1}  {item['name']}  {item['hex']}  {item['percentage']}%  ({item['pixel_count']} px)"
            canvas = self._draw_chinese_text(canvas, info_text, (48, y_offset + 1),
                                              font_size=14, color=(220, 220, 220))
            y_offset += info_height

        output_path = os.path.join(self.output_dir, 'color_bar.png')
        cv2.imwrite(output_path, canvas)
        print(f"[信息] 色条图已保存: {output_path}")
        return canvas

    def generate_pie_chart(self, size=500):
        """
        生成颜色占比饼图
        :param size: 饼图尺寸
        :return: 饼图图像
        """
        legend_width = 200
        canvas = np.ones((size, size + legend_width, 3), dtype=np.uint8) * 45

        center = (size // 2, size // 2)
        radius = int(size * 0.38)

        # 绘制标题（中文）
        canvas = self._draw_chinese_text(canvas, "颜色占比饼图", (15, 8), font_size=20)

        # 逐个绘制扇形
        start_angle = 0
        for i, item in enumerate(self.color_results):
            sweep_angle = int(360 * item['proportion'])
            if sweep_angle == 0:
                sweep_angle = 1  # 至少1度保证可见

            end_angle = start_angle + sweep_angle
            cv2.ellipse(canvas, center, (radius, radius),
                        0, start_angle, end_angle, item['bgr'], -1)

            # 绘制扇形边线
            cv2.ellipse(canvas, center, (radius, radius),
                        0, start_angle, end_angle, (80, 80, 80), 2)

            start_angle = end_angle

        # 绘制圆边框
        cv2.circle(canvas, center, radius, (150, 150, 150), 2)

        # 绘制图例（中文）
        legend_x = size + 15
        legend_y = 50
        for i, item in enumerate(self.color_results):
            cv2.rectangle(canvas,
                          (legend_x, legend_y),
                          (legend_x + 16, legend_y + 16),
                          item['bgr'], -1)
            text = f"{item['name']} {item['percentage']}%"
            canvas = self._draw_chinese_text(canvas, text, (legend_x + 24, legend_y),
                                              font_size=14, color=(220, 220, 220))
            legend_y += 28

        output_path = os.path.join(self.output_dir, 'pie_chart.png')
        cv2.imwrite(output_path, canvas)
        print(f"[信息] 饼图已保存: {output_path}")
        return canvas

    def generate_html_report(self):
        """生成完整的HTML格式分析报告（含图片展示）"""

        # 图片转base64嵌入
        original_b64 = self._img_to_base64(self.original_img) if self.original_img is not None else ""
        clustered_b64 = self._img_to_base64(self.clustered_img) if self.clustered_img is not None else ""

        rows = ""
        for i, item in enumerate(self.color_results):
            rows += f"""
            <tr>
                <td>{i + 1}</td>
                <td>
                    <span class="color-swatch"
                          style="background:{item['hex']};"></span>
                    {item['name']}
                </td>
                <td><code>{item['hex']}</code></td>
                <td>({item['rgb'][0]}, {item['rgb'][1]}, {item['rgb'][2]})</td>
                <td><strong>{item['percentage']}%</strong></td>
                <td>{item['pixel_count']:,}</td>
            </tr>"""

        proportion_bars = ""
        for item in self.color_results:
            proportion_bars += f"""
            <div class="bar-segment" style="
                width:{item['percentage']}%;
                background:{item['hex']};">
                <span>{item['percentage']}%</span>
            </div>"""

        # 图片展示区
        images_section = ""
        if original_b64 or clustered_b64:
            images_section = '<h2>图像展示</h2><div class="img-row">'
            if original_b64:
                images_section += f"""
                <div class="img-card">
                    <img src="data:image/png;base64,{original_b64}" alt="原图">
                    <p>原始图像</p>
                </div>"""
            if clustered_b64:
                images_section += f"""
                <div class="img-card">
                    <img src="data:image/png;base64,{clustered_b64}" alt="聚类图">
                    <p>聚类结果</p>
                </div>"""
            images_section += '</div>'

        # 色卡展示
        swatch_section = '<h2>主要色卡</h2><div class="swatch-row">'
        for item in self.color_results:
            swatch_section += f"""
            <div class="swatch-card">
                <div class="swatch-block" style="background:{item['hex']};"></div>
                <p>{item['name']}</p>
                <p>{item['hex']}</p>
                <p>{item['percentage']}%</p>
            </div>"""
        swatch_section += '</div>'

        html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>色块占比分析报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', '微软雅黑', 'PingFang SC', 'SimHei', sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            padding: 40px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 28px;
            margin-bottom: 8px;
            color: #e0e0e0;
        }
        h2 {
            margin: 30px 0 10px;
            font-size: 18px;
            color: #ccc;
            border-left: 4px solid #e8c547;
            padding-left: 10px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        .bar-chart {
            display: flex;
            height: 50px;
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }
        .bar-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.6);
            min-width: 20px;
            transition: all 0.3s;
        }
        .bar-segment:hover { filter: brightness(1.2); }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th {
            background: #16213e;
            padding: 12px 10px;
            text-align: left;
            font-size: 13px;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #2a2a4a;
            font-size: 13px;
        }
        tr:hover { background: rgba(255,255,255,0.03); }
        .color-swatch {
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 3px;
            vertical-align: middle;
            margin-right: 6px;
            border: 1px solid #555;
        }
        code {
            background: #16213e;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        .info-box {
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 14px;
            line-height: 1.8;
        }
        .img-row {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .img-card {
            flex: 1;
            min-width: 250px;
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
            text-align: center;
        }
        .img-card img {
            width: 100%;
            display: block;
        }
        .img-card p {
            padding: 10px;
            font-size: 14px;
            color: #aaa;
        }
        .swatch-row {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .swatch-card {
            background: #16213e;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            width: 120px;
        }
        .swatch-block {
            width: 80px;
            height: 80px;
            border-radius: 8px;
            margin: 0 auto 10px;
            border: 2px solid #333;
        }
        .swatch-card p {
            font-size: 12px;
            color: #bbb;
            margin: 2px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>色块占比分析报告</h1>
        <p class="subtitle">Color Block Proportion Analysis Report</p>

        <div class="info-box">
            <strong>图像文件：</strong>""" + str(self.image_info.get('file_name', 'N/A')) + """<br>
            <strong>图像尺寸：</strong>""" + str(self.image_info.get('width', 0)) + """ x """ + str(self.image_info.get('height', 0)) + """<br>
            <strong>总像素数：</strong>""" + f"{self.image_info.get('total_pixels', 0):,}" + """<br>
            <strong>提取颜色数：</strong>""" + str(len(self.color_results)) + """
        </div>

        """ + images_section + """

        <h2>占比分布</h2>
        <div class="bar-chart">""" + proportion_bars + """</div>

        """ + swatch_section + """

        <h2>详细数据</h2>
        <table>
            <thead>
                <tr>
                    <th>序号</th>
                    <th>颜色</th>
                    <th>HEX</th>
                    <th>RGB</th>
                    <th>占比</th>
                    <th>像素数</th>
                </tr>
            </thead>
            <tbody>""" + rows + """</tbody>
        </table>
    </div>
</body>
</html>"""

        output_path = os.path.join(self.output_dir, 'report.html')
        # 用 UTF-8 BOM 写入，确保浏览器识别中文编码
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write(html)

        print(f"[信息] HTML报告已保存: {output_path}")
        return output_path

    def show_live_preview(self, original_img, clustered_img):
        """
        实时显示原图与聚类结果的对比
        :param original_img: 原始图像
        :param clustered_img: 聚类后的图像
        """
        h, w = original_img.shape[:2]
        preview_w = min(640, w)
        preview_h = int(h * preview_w / w)

        display_orig = cv2.resize(original_img, (preview_w, preview_h))
        display_cluster = cv2.resize(clustered_img, (preview_w, preview_h))

        # 拼接两图
        separator = np.ones((preview_h, 3, 3), dtype=np.uint8) * 200
        combined = np.hstack([display_orig, separator, display_cluster])

        # 用PIL显示（支持中文标题）
        pil_combined = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        pil_combined.show(title="原始图像 | 聚类结果")

    def save_all(self, original_img, clustered_img):
        """一键保存所有输出文件"""
        self.generate_color_bar()
        self.generate_pie_chart()
        self.generate_html_report()

        # 保存聚类结果图
        cv2.imwrite(os.path.join(self.output_dir, 'clustered.png'), clustered_img)
        print(f"[信息] 所有结果已保存至 '{self.output_dir}/' 目录")
