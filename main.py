"""
main.py
色块占比识别系统 - 主程序入口
功能：协调图像加载、颜色分析和结果可视化
"""

import os
import sys
import subprocess
from image_loader import ImageLoader
from color_analyzer import ColorAnalyzer
from result_visualizer import ResultVisualizer


def select_image_file():
    """
    弹出文件选择对话框让用户选择图片
    :return: 选择的文件路径，取消则返回 None
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # 创建并隐藏主窗口
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # 置顶显示

        file_path = filedialog.askopenfilename(
            title="请选择要分析的图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("所有文件", "*.*"),
            ]
        )

        root.destroy()
        return file_path if file_path else None

    except ImportError:
        print("[警告] 未安装 tkinter，回退为手动输入路径")
        path = input("请输入图像文件路径: ").strip().strip('"')
        return path if path else None


def main():
    print("=" * 50)
    print("    色块占比识别系统 v1.0")
    print("    Color Block Proportion Analyzer")
    print("=" * 50)

    # ========== 1. 弹窗选择图片 ==========
    print("\n正在打开文件选择对话框...")
    image_path = select_image_file()

    if not image_path:
        print("[信息] 未选择文件，程序退出。")
        return

    print(f"[信息] 已选择: {image_path}")

    # ========== 2. 加载与预处理图像 ==========
    print("\n--- 第1步: 加载图像 ---")
    loader = ImageLoader()
    if not loader.load_image(image_path):
        print("程序终止。")
        return

    # 缩放和降噪
    loader.resize(target_width=800)
    loader.apply_blur(kernel_size=5)

    # 显示图像信息
    info = loader.get_image_info()
    print(f"\n图像信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # ========== 3. 颜色聚类分析 ==========
    print("\n--- 第2步: 颜色分析 ---")
    num_colors = 5
    try:
        user_input = input("请输入要提取的颜色数量 (默认5): ").strip()
        if user_input:
            num_colors = int(user_input)
    except ValueError:
        num_colors = 5

    analyzer = ColorAnalyzer(loader.get_image_data())
    results = analyzer.analyze(num_colors=num_colors)

    if results is None:
        print("颜色分析失败，程序终止。")
        return

    # 打印终端报告
    analyzer.print_report()

    # 获取聚类结果图
    clustered_image = analyzer.get_label_image()

    # ========== 4. 可视化输出 ==========
    print("--- 第3步: 生成可视化报告 ---")
    visualizer = ResultVisualizer(
        image_info=info,
        color_results=results,
        original_img=loader.get_original_data(),
        clustered_img=clustered_image,
        output_dir='output'
    )

    visualizer.save_all(
        original_img=loader.get_original_data(),
        clustered_img=clustered_image
    )

    # ========== 5. 自动打开报告 ==========
    report_path = os.path.abspath(os.path.join('output', 'report.html'))
    print(f"\n[信息] 报告路径: {report_path}")
    try:
        if sys.platform == 'win32':
            os.startfile(report_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', report_path])
        else:
            subprocess.run(['xdg-open', report_path])
        print("[信息] 已在浏览器中打开报告")
    except Exception:
        print(f"[信息] 请手动打开: {report_path}")

    # ========== 6. 实时预览 ==========
    try:
        show = input("\n是否显示实时预览？(y/n, 默认y): ").strip().lower()
        if show != 'n':
            visualizer.show_live_preview(
                loader.get_original_data(),
                clustered_image
            )
    except KeyboardInterrupt:
        pass

    print("\n分析完成！")


if __name__ == '__main__':
    main()
