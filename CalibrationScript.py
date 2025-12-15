import cv2
import numpy as np
import os
import glob
import argparse
import math
import yaml
from datetime import datetime


def calculate_fov(camera_matrix, image_size, fisheye_mode=True):
    """
    计算相机视场角（水平、垂直、对角），针对鱼眼镜头进行优化

    参数:
    camera_matrix: 相机内参矩阵 (3x3)
    image_size: 图像分辨率 (width, height)
    fisheye_mode: 是否为鱼眼镜头模式，默认True

    返回:
    hfov, vfov, dfov: 水平、垂直、对角视场角（单位：度）
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    width, height = image_size

    if fisheye_mode:
        # 鱼眼等距投影：计算左右/上下视角之和（精确版）
        theta_left = math.asin(cx / fx)
        theta_right = math.asin((width - 1 - cx) / fx)
        hfov = math.degrees(theta_left + theta_right)

        theta_bottom = math.asin(cy / fy)
        theta_top = math.asin((height - 1 - cy) / fy)
        vfov = math.degrees(theta_bottom + theta_top)

        # 对于对角视场角，可以简单地用勾股定理结合hfov和vfov来近似
        diagonal = math.sqrt(width ** 2 + height ** 2)
        diagonal_angle = math.atan2(diagonal, max(fx, fy))
        dfov = math.degrees(2 * diagonal_angle)
    else:
        # 普通镜头的标准计算方法
        hfov = 2 * math.atan(width / (2 * fx)) * (180 / math.pi)
        vfov = 2 * math.atan(height / (2 * fy)) * (180 / math.pi)
        diagonal = math.sqrt(width ** 2 + height ** 2)
        dfov = 2 * math.atan(diagonal / (2 * math.sqrt(fx ** 2 + fy ** 2))) * (180 / math.pi)

    return hfov, vfov, dfov


def calibrate_normal_camera(objpoints, imgpoints, image_size):
    """
    使用普通相机模型进行标定
    """
    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # 计算平均重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    return ret, mtx, dist, rvecs, tvecs, mean_error


def calibrate_fisheye_camera(objpoints, imgpoints, image_size):
    """
    使用鱼眼相机模型进行标定
    """
    # 为鱼眼标定准备数据 - 确保数据类型和形状正确
    objpoints_fisheye = []
    imgpoints_fisheye = []

    # 准备3D点和2D点，确保正确的数据类型
    for i in range(len(objpoints)):
        # 3D点必须是float64类型，形状为(N, 1, 3)
        objpoints_fisheye.append(np.array(objpoints[i], dtype=np.float64).reshape(-1, 1, 3))
        # 2D点必须是float64或float32类型，形状为(N, 1, 2)
        imgpoints_fisheye.append(np.array(imgpoints[i], dtype=np.float64).reshape(-1, 1, 2))

    # 初始化内参矩阵
    mtx = np.zeros((3, 3))
    mtx[0, 0] = 1  # 初始焦距
    mtx[1, 1] = 1
    mtx[2, 2] = 1

    # 预分配畸变系数
    dist = np.zeros((4, 1))

    # 鱼眼标定标志
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

    try:
        # 执行鱼眼标定
        num_frames = len(objpoints_fisheye)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(num_frames)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(num_frames)]

        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_fisheye, imgpoints_fisheye, image_size,
            mtx, dist, rvecs, tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        # 计算平均重投影误差
        total_error = 0
        for i in range(len(objpoints_fisheye)):
            # 重投影
            projected_points, _ = cv2.fisheye.projectPoints(
                objpoints_fisheye[i], rvecs[i], tvecs[i], mtx, dist
            )
            # 计算误差
            error = cv2.norm(imgpoints_fisheye[i], projected_points, cv2.NORM_L2) / len(imgpoints_fisheye[i])
            total_error += error
        mean_error = total_error / len(objpoints_fisheye)

        return ret, mtx, dist, rvecs, tvecs, mean_error

    except cv2.error as e:
        print(f"鱼眼标定失败: {e}")
        print("尝试使用普通相机模型进行标定...")
        return calibrate_normal_camera(objpoints, imgpoints, image_size)


def save_calibration_to_yaml(output_file, camera_matrix, dist_coeffs, reprojection_error,
                             hfov, vfov, dfov, image_size, lens_type, calibration_params):
    """
    将标定结果保存为YAML格式
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 准备YAML数据结构
    yaml_data = {
        'calibration_date': str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        'lens_type': lens_type,
        'image_resolution': {
            'width': int(image_size[0]),
            'height': int(image_size[1])
        },
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [float(x) for x in camera_matrix.flatten()]
        },
        'distortion_coefficients': {
            'count': len(dist_coeffs.flatten()),
            'data': [float(x) for x in dist_coeffs.flatten()]
        },
        'field_of_view': {
            'horizontal': float(hfov),
            'vertical': float(vfov),
            'diagonal': float(dfov)
        },
        'reprojection_error': {
            'rms': float(reprojection_error),
            'unit': 'pixels'
        },
        'calibration_parameters': {
            'chessboard_size': {
                'columns': calibration_params['chessboard_size'][0],
                'rows': calibration_params['chessboard_size'][1]
            },
            'square_size_mm': calibration_params['square_size'],
            'used_images_count': calibration_params['used_images_count'],
            'total_images_count': calibration_params['total_images_count']
        },
        'notes': [
            'HFOV: Horizontal Field of View (水平视场角)',
            'VFOV: Vertical Field of View (垂直视场角)',
            'DFOV: Diagonal Field of View (对角视场角)',
            'RMS error < 1.0 is considered good for most applications',
            'For fisheye cameras, error < 2.0 may be acceptable due to extreme distortion'
        ]
    }

    # 保存为YAML文件
    try:
        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"\n✓ 标定结果已保存为YAML格式: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        print(f"保存YAML文件时出错: {e}")
        return False


def calibrate_camera(image_path, image_resolution, chessboard_size, square_size, lens_type='normal', image_prefix='left-'):
    """
    相机标定函数 - 支持普通和鱼眼镜头模式

    参数:
    image_path: 图片存储路径
    image_resolution: 图片分辨率 (width, height)
    chessboard_size: 棋盘格内角点数量 (cols, rows)
    square_size: 棋盘格单个方格的实际尺寸(mm)
    lens_type: 镜头类型 ('normal' 或 'fisheye')

    返回:
    标定结果，包含重投影误差、内参矩阵、畸变系数等
    """
    print(f"\n使用 '{'鱼眼镜头' if lens_type == 'fisheye' else '普通镜头'}' 模型进行标定")

    # 准备世界坐标系中的3D点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 应用实际尺寸

    # 存储3D点和2D点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    # 获取所有符合命名规则的图片
    pattern = os.path.join(image_path, f"{image_prefix}-*.png")
    images = sorted(glob.glob(pattern))

    # 如果没有找到图片，尝试其他可能的命名格式
    if not images:
        print(f"未找到符合 '{image_prefix}-*.png' 格式的图片，尝试其他可能的命名格式...")
        # 尝试 left*.png (没有连字符)
        pattern2 = os.path.join(image_path, f"{image_prefix}*.png")
        images = sorted(glob.glob(pattern2))

        if not images:
            # 尝试 *.png (所有png文件)
            pattern3 = os.path.join(image_path, "*.png")
            all_pngs = sorted(glob.glob(pattern3))
            # 筛选包含"{image_prefix}"的文件名
            images = [img for img in all_pngs if f"{image_prefix}" in os.path.basename(img).lower()]

    if not images:
        print(f"\n错误: 在路径 '{image_path}' 中未找到符合命名规则的图片")
        print(f"搜索模式: {pattern}")
        print("请检查:")
        print(f"1. 路径 '{image_path}' 是否正确")
        print("2. 图片是否确实存在于该路径")
        print(f"3. 图片是否符合 '{image_prefix}-*.png' 命名格式")
        # 列出目录内容帮助诊断
        if os.path.exists(image_path):
            print(f"\n目录 '{image_path}' 中的内容:")
            for item in os.listdir(image_path):
                print(f"  - {item}")
        return None

    print(f"\n找到 {len(images)} 张符合命名规则的标定图片:")
    for i, img_path in enumerate(images[:5], 1):  # 只显示前5个
        print(f"  {i}. {os.path.basename(img_path)}")
    if len(images) > 5:
        print(f"  ... (共 {len(images)} 张图片)")

    print("\n开始检测棋盘格角点...")

    # 创建显示窗口
    cv2.namedWindow('棋盘格角点检测', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('棋盘格角点检测', 800, 600)

    valid_images = 0
    total_images = len(images)
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"警告: 无法读取图片: {fname} (可能是损坏的文件或不支持的格式)")
            continue

        # 检查并调整图片尺寸
        actual_height, actual_width = img.shape[:2]
        if (actual_width, actual_height) != image_resolution:
            print(
                f"调整图片 {os.path.basename(fname)} 尺寸: {actual_width}x{actual_height} -> {image_resolution[0]}x{image_resolution[1]}")
            img = cv2.resize(img, image_resolution)

        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None, cv2.CALIB_CB_EXHAUSTIVE)

        if ret:
            valid_images += 1
            objpoints.append(objp.copy())

            # 亚像素级精确化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_sub)

            # 绘制并显示角点
            cv2.drawChessboardCorners(img, chessboard_size, corners_sub, ret)
            cv2.imshow('棋盘格角点检测', img)
            key = cv2.waitKey(300)  # 显示0.3秒
            if key == 27:  # ESC键
                cv2.destroyAllWindows()
                print("\n用户中断标定过程")
                return None

            print(f"✓ 成功检测图片 {os.path.basename(fname)} 中的角点")
        else:
            print(
                f"✗ 在图片 {os.path.basename(fname)} 中未找到完整的棋盘格 ({chessboard_size[0]}x{chessboard_size[1]} 内角点)")

    cv2.destroyAllWindows()

    if valid_images < 3:
        print(f"\n错误: 有效标定图片数量不足 ({valid_images}张)，至少需要3张图片进行标定")
        print("可能原因:")
        print("- 棋盘格内角点数量设置不正确")
        print("- 图片质量不佳或棋盘格不完整")
        print("- 棋盘格尺寸与设置不符")
        return None

    print(f"\n✓ 使用 {valid_images}/{total_images} 张有效图片进行标定")

    # 获取图像尺寸 (OpenCV需要(width, height))
    h, w = gray.shape[:2]
    image_size = (w, h)

    # 根据镜头类型选择标定函数
    if lens_type == 'fisheye':
        print("\n使用鱼眼镜头模型进行标定 (cv::fisheye::calibrate)...")
        result = calibrate_fisheye_camera(objpoints, imgpoints, image_size)
    else:
        print("\n使用普通镜头模型进行标定 (cv::calibrateCamera)...")
        result = calibrate_normal_camera(objpoints, imgpoints, image_size)

    if result is None:
        print("标定过程失败")
        return None

    ret, mtx, dist, rvecs, tvecs, mean_error = result

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, image_size, valid_images, total_images


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='相机内参标定工具 (普通/鱼眼镜头) + YAML输出',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--image_path', type=str, default=r'D:\Schen\CalibrationScript\calibrationdata_2',
                        help='标定图片存储路径\n(默认: D:\\Schen\\CalibrationScript\\calibrationdata_0)')
    parser.add_argument('--image_prefix', type=str, default=r'left-',
                        help='标定图片名称的的统一前缀')
    parser.add_argument('--width', type=int, default=1280,
                        help='图片宽度(像素)\n(默认: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='图片高度(像素)\n(默认: 720)')
    parser.add_argument('--pattern_cols', type=int, default=11,
                        help='棋盘格内角点列数(宽度方向)\n(默认: 11)')
    parser.add_argument('--pattern_rows', type=int, default=8,
                        help='棋盘格内角点行数(高度方向)\n(默认: 8)')
    parser.add_argument('--square_size', type=float, default=30.0,
                        help='棋盘格方格实际尺寸(mm)\n(默认: 30.0)')
    parser.add_argument('--lens_type', type=str, choices=['normal', 'fisheye'], default='fisheye',
                        help='镜头类型:\n'
                             '  normal: 普通镜头 (标准针孔模型)\n'
                             '  fisheye: 广角/鱼眼镜头 (专用鱼眼模型)\n'
                             '(默认: normal)')
    parser.add_argument('--output_yaml', type=str, default='calibration_result_2.yaml',
                        help='标定结果YAML保存文件\n(默认: calibration_result_0.yaml)')

    args = parser.parse_args()

    # 获取参数
    image_path = args.image_path
    image_prefix = args.image_prefix
    image_resolution = (args.width, args.height)
    chessboard_size = (args.pattern_cols, args.pattern_rows)
    square_size = args.square_size
    lens_type = args.lens_type
    output_yaml = args.output_yaml

    print("=" * 70)
    print(f"相机标定工具 - {'普通镜头' if lens_type == 'normal' else '鱼眼镜头'} 模式")
    print("=" * 70)
    print("参数设置:")
    print(f"- 图片路径: {image_path}")
    print(f"- 图片名称前缀: {image_prefix}")
    print(f"- 目标分辨率: {image_resolution[0]}x{image_resolution[1]} 像素")
    print(f"- 棋盘格内角点: {chessboard_size[0]}x{chessboard_size[1]}")
    print(f"- 方格实际尺寸: {square_size} mm")
    print(f"- 镜头类型: {'普通镜头 (标准模型)' if lens_type == 'normal' else '广角/鱼眼镜头 (专用模型)'}")
    print(f"- 结果保存至: {output_yaml}")
    print("-" * 70)

    # 检查路径
    if not os.path.exists(image_path):
        print(f"\n错误: 路径 '{image_path}' 不存在")
        # 尝试提供可能的正确路径
        possible_paths = [
            os.path.dirname(image_path),
            os.path.join(os.path.dirname(os.path.dirname(image_path)), 'calibrationdata_0'),
            os.path.join(os.path.dirname(image_path), 'calibrationdata_0')
        ]

        print("\n可能的正确路径:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✓ {path}")
            else:
                print(f"  ✗ {path} (不存在)")

        # 列出当前目录内容
        current_dir = os.getcwd()
        print(f"\n当前工作目录: {current_dir}")
        if os.path.exists(current_dir):
            print("当前目录内容:")
            for item in os.listdir(current_dir):
                print(f"  - {item}")

        return

    print(f"\n✓ 路径 '{image_path}' 存在，开始处理...")

    # 执行标定
    result = calibrate_camera(image_path, image_resolution, chessboard_size, square_size, lens_type, image_prefix)

    if result is None:
        print("\n" + "=" * 70)
        print("标定过程失败，请检查上述错误信息并修正后重试")
        print("=" * 70)
        return

    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, actual_image_size, valid_images, total_images = result

    # 打印标定结果
    print("\n" + "=" * 70)
    print("✓ 标定成功完成!")
    print(f"重投影误差: {ret:.4f} 像素 (值越小表示标定质量越高，通常<1.0为佳)")
    print("\n相机内参矩阵 (Camera Matrix):")
    print(mtx)
    print("\n畸变系数 (Distortion Coefficients):")

    if lens_type == 'fisheye':
        print("鱼眼模型使用4个畸变参数 (k1, k2, k3, k4):")
        # 确保dist是4x1的形状
        if dist.shape[0] > 4:
            dist = dist[:4]
    else:
        print("普通模型使用5个畸变参数 (k1, k2, p1, p2, k3):")
    print(dist.flatten())
    print("=" * 70)

    # 计算视场角
    print("\n" + "-" * 70)
    print("计算相机视场角 (Field of View)...")
    hfov, vfov, dfov = calculate_fov(mtx, actual_image_size, lens_type == 'fisheye')

    print("\n相机视场角 (FOV) 计算结果:")
    print(f"  水平视场角 (HFOV): {hfov:.2f}°")
    print(f"  垂直视场角 (VFOV): {vfov:.2f}°")
    print(f"  对角视场角 (DFOV): {dfov:.2f}°")
    print("-" * 70)

    # 视场角应用场景说明
    print("\n视场角应用场景参考:")
    print(f"  适用于 {actual_image_size[0]}x{actual_image_size[1]} 分辨率的图像")
    if hfov < 60:
        print("  - 窄视场相机，适合远距离目标跟踪、望远观测")
    elif hfov < 90:
        print("  - 标准视场相机，适合一般监控、机器视觉应用")
    elif hfov < 120:
        print("  - 广角相机，适合环境监测、大范围场景捕捉")
    else:
        print("  - 超广角/鱼眼相机，适合全景拍摄、VR应用")
        print("  - 注意: 鱼眼镜头会产生显著的桶形畸变，需使用专用模型校正")
    print("-" * 70)

    # 准备标定参数用于YAML保存
    calibration_params = {
        'chessboard_size': chessboard_size,
        'square_size': square_size,
        'used_images_count': valid_images,
        'total_images_count': total_images
    }

    # 保存标定结果为YAML
    save_calibration_to_yaml(output_yaml, mtx, dist, ret, hfov, vfov, dfov,
                             actual_image_size, lens_type, calibration_params)

    # 标定质量评估
    print("\n标定质量评估:")
    if ret < 0.5:
        print("  ✓ 优秀: 重投影误差 < 0.5 像素")
    elif ret < 1.0:
        print("  ✓ 良好: 重投影误差 < 1.0 像素")
    elif ret < 2.0:
        print("  △ 一般: 重投影误差 < 2.0 像素，可能需要更多/更好的标定图片")
    else:
        print("  ✗ 较差: 重投影误差 >= 2.0 像素，建议重新标定或检查参数设置")

    if lens_type == 'fisheye' and ret > 2.0:
        print("  ⚠ 鱼眼镜头标定误差较大，考虑:")
        print("    - 检查镜头是否确实是鱼眼类型")
        print("    - 增加标定图片数量和多样性")
        print("    - 调整棋盘格大小或位置，覆盖更多图像区域，特别是边缘区域")
        print("    - 确保棋盘格在图像中不要过于扭曲或靠近边缘")


if __name__ == "__main__":
    # 确保PyYAML库已安装
    try:
        import yaml
    except ImportError:
        print("警告: PyYAML库未安装，将无法保存YAML格式结果")
        print("请安装PyYAML: pip install pyyaml")
        exit(1)

    main()


