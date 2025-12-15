import cv2
import numpy as np
import os
import argparse
import yaml
import time
from datetime import datetime


def load_calibration(calibration_file):
    """加载相机标定参数"""
    try:
        with open(calibration_file, 'r') as f:
            calibration_data = yaml.safe_load(f)
        # 提取必要的参数
        camera_matrix = np.array(calibration_data['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(calibration_data['distortion_coefficients']['data'])
        lens_type = calibration_data['lens_type']
        image_resolution = (calibration_data['image_resolution']['width'],
                            calibration_data['image_resolution']['height'])
        return camera_matrix, dist_coeffs, lens_type, image_resolution
    except Exception as e:
        print(f"加载标定文件失败: {e}")
        return None, None, None, None


def undistort_image(frame, camera_matrix, dist_coeffs, lens_type, image_resolution):
    """根据镜头类型进行畸变矫正"""
    if lens_type == 'fisheye':
        # 鱼眼镜头矫正
        h, w = image_resolution
        # 使用OpenCV的鱼眼矫正函数
        try:
            # 创建映射
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, np.eye(3), camera_matrix,
                (w, h), cv2.CV_16SC2
            )
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            return undistorted
        except Exception as e:
            print(f"鱼眼矫正失败: {e}")
            # 如果fisheye模块不可用，尝试使用普通矫正
            return cv2.undistort(frame, camera_matrix, dist_coeffs)
    else:
        # 普通镜头矫正
        return cv2.undistort(frame, camera_matrix, dist_coeffs)


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='多相机实时拍摄与保存工具',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--cameras', type=str, nargs='+', default=['rtsp://127.0.0.1:8554/stream1'],
                        help='相机连接参数，可以是RTSP URL或USB设备索引\n'
                             '示例: --cameras rtsp://127.0.0.1:8554/stream1 0 1\n'
                             '默认: rtsp://127.0.0.1:8554/stream1\n'
                             '注意: USB相机使用设备索引（如0,1,2），不是串口')
    parser.add_argument('--save_path', type=str, default='captured_images',
                        help='保存图片的路径 (默认: captured_images)')
    parser.add_argument('--save_format', type=str, choices=['png', 'jpg'], default='png',
                        help='保存图片的格式 (png或jpg, 默认: png)')
    parser.add_argument('--calibration_file', type=str, default=None,
                        help='相机标定YAML文件路径，用于畸变矫正 (可选)')
    parser.add_argument('--display_original', action='store_true',
                        help='显示原始图像 (不进行畸变矫正)')
    parser.add_argument('--display_undistorted', action='store_true',
                        help='显示畸变矫正后的图像 (需要提供calibration_file)')
    args = parser.parse_args()

    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)

    # 检查是否需要加载标定文件
    camera_matrix = None
    dist_coeffs = None
    lens_type = None
    image_resolution = None
    if args.calibration_file:
        camera_matrix, dist_coeffs, lens_type, image_resolution = load_calibration(args.calibration_file)
        if camera_matrix is None or dist_coeffs is None:
            print("警告: 无法加载标定文件，畸变矫正功能将不可用")
            args.display_undistorted = False

    # 初始化相机
    cap_list = []
    for cam in args.cameras:
        # 尝试解析为RTSP URL
        if cam.startswith('rtsp://'):
            cap = cv2.VideoCapture(cam)
            print(f"正在连接RTSP相机: {cam}")
        else:
            # 假设是USB相机设备索引
            try:
                cam_index = int(cam)
                cap = cv2.VideoCapture(cam_index)
                print(f"正在连接USB相机 (设备索引: {cam_index})")
            except ValueError:
                print(f"错误: 无效的相机参数 '{cam}'，应为RTSP URL或整数设备索引")
                continue

        if not cap.isOpened():
            print(f"错误: 无法打开相机 '{cam}'")
            continue
        cap_list.append((cap, cam))

    if not cap_list:
        print("错误: 没有成功连接到任何相机")
        return

    # 创建窗口
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', 1280, 720)

    # 用于保存的键
    save_key = ord('s')
    quit_key = ord('q')
    next_key = ord('n')
    prev_key = ord('p')

    # 当前相机索引
    current_cam_index = 0

    # 用于记录保存的图片数量
    save_counter = 0

    # 显示信息
    print("\n相机控制:")
    print("  - 按 's' 保存当前帧到文件")
    print("  - 按 'q' 退出程序")
    print("  - 按 'n' 切换到下一个相机")
    print("  - 按 'p' 切换到上一个相机")
    print(f"  - 保存路径: {os.path.abspath(args.save_path)}")
    print(f"  - 保存格式: {args.save_format}")
    print(f"  - 显示原始图像: {'是' if args.display_original else '否'}")
    print(f"  - 显示畸变矫正图像: {'是' if args.display_undistorted else '否'}")

    # 主循环
    while True:
        # 获取当前相机
        cap, cam_name = cap_list[current_cam_index]

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法从相机 '{cam_name}' 读取帧")
            # 尝试重新打开相机
            cap.release()
            if cam_name.startswith('rtsp://'):
                cap = cv2.VideoCapture(cam_name)
            else:
                try:
                    cam_index = int(cam_name)
                    cap = cv2.VideoCapture(cam_index)
                except:
                    cap = None
            if cap and cap.isOpened():
                print(f"已重新打开相机 '{cam_name}'")
                cap_list[current_cam_index] = (cap, cam_name)
            else:
                print(f"无法重新打开相机 '{cam_name}'，跳过")
                current_cam_index = (current_cam_index + 1) % len(cap_list)
                continue

        # 选择显示的图像
        if args.display_undistorted and camera_matrix is not None and dist_coeffs is not None:
            # 进行畸变矫正
            undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs, lens_type, image_resolution)
            display_frame = undistorted_frame
        elif args.display_original:
            display_frame = frame
        else:
            # 默认显示原始图像
            display_frame = frame

        # 显示相机名称
        cv2.putText(display_frame, f"Camera: {cam_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示当前相机索引
        cv2.putText(display_frame, f"Camera Index: {current_cam_index + 1}/{len(cap_list)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示保存信息
        cv2.putText(display_frame, f"Save Count: {save_counter}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示保存格式
        cv2.putText(display_frame, f"Save Format: {args.save_format}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示当前帧
        cv2.imshow('Camera Feed', display_frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        # 保存图片
        if key == save_key:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(args.save_path,
                                    f"camera_{cam_name.replace(':', '_')}_{timestamp}.{args.save_format}")

            # 保存图片
            if cv2.imwrite(filename, display_frame):
                save_counter += 1
                print(f"✓ 已保存图片: {filename}")
            else:
                print(f"✗ 保存图片失败: {filename}")

        # 退出程序
        elif key == quit_key:
            break

        # 切换到下一个相机
        elif key == next_key:
            current_cam_index = (current_cam_index + 1) % len(cap_list)
            print(f"切换到相机: {cap_list[current_cam_index][1]}")

        # 切换到上一个相机
        elif key == prev_key:
            current_cam_index = (current_cam_index - 1) % len(cap_list)
            print(f"切换到相机: {cap_list[current_cam_index][1]}")

    # 释放资源
    for cap, _ in cap_list:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()