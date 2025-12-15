# 相机内参标定工具 (普通/鱼眼镜头)

## 1. 简介

本工具是一个功能完善的相机内参标定程序，支持普通相机和鱼眼/广角镜头的标定。它能够从标定图像中提取棋盘格角点，计算相机内参矩阵和畸变系数，并将结果保存为结构化的 YAML 格式文件。该工具特别优化了鱼眼镜头的标定流程和视场角计算方法，适用于计算机视觉、机器人导航、增强现实等多种应用场景。







## 脚本功能说明

1. **支持多种相机类型**:
   - RTSP网络相机：使用`rtsp://`前缀连接
   - USB相机：通过设备索引（如0,1,2）连接（注：USB相机不是通过串口连接，而是通过设备索引）
2. **多相机支持**:
   - 可以同时连接多个相机，通过`--cameras`参数指定
   - 例如：`--cameras rtsp://127.0.0.1:8554/stream1 0 1`
3. **实时显示与控制**:
   - 每个相机的视频流在同一个窗口中显示
   - 使用键盘控制：
     - `s`: 保存当前帧
     - `q`: 退出程序
     - `n`: 切换到下一个相机
     - `p`: 切换到上一个相机
4. **畸变矫正**:
   - 通过`--calibration_file`参数加载YAML标定文件
   - 支持鱼眼相机和普通相机的畸变矫正
   - 通过`--display_undistorted`启用畸变矫正显示
5. **图片保存**:
   - 保存路径：`--save_path`（默认`captured_images`）
   - 保存格式：`--save_format`（默认`png`，可选`jpg`）
   - 文件名包含时间戳和相机标识

## 使用示例

连接一个RTSP相机并保存PNG图片：

```
python camera_capture.py --cameras rtsp://192.168.1.100:8554/stream1 --save_format png
```

连接两个USB相机（设备索引0和1）并启用畸变矫正：

```
python camera_capture.py --cameras 0 1 --calibration_file calibration.yaml --display_undistorted
```

连接RTSP相机和USB相机（设备索引0）：

```
1python camera_capture.py --cameras rtsp://192.168.1.100:8554/stream1 0 --save_path ./my_images --save_format jpg
```

## 注意事项

1. USB相机连接：脚本使用设备索引（如0,1,2）连接USB相机，而不是串口。如果您的USB相机无法通过设备索引访问，可能需要检查相机驱动或系统设置。
2. 畸变矫正：需要提供有效的YAML标定文件，格式与您提供的示例一致。
3. 鱼眼相机：当`lens_type`为`fisheye`时，脚本会使用专用的鱼眼畸变矫正函数。







## 2. 功能特点

- ✅**支持多种类型的相机的拍摄**:
  - RTSP网络相机：使用`rtsp://`前缀连接
  - USB相机：通过设备索引（如0,1,2）连接
  - 多相机支持，可以同时连接多个相机，通过`--cameras`参数指定，例如：`--cameras rtsp://127.0.0.1:8554/stream1 0 1`
- ✅**实时显示与控制**:
  - 每个相机的视频流在同一个窗口中显示
  - 使用键盘控制：
    - `s`: 保存当前帧
    - `q`: 退出程序
    - `n`: 切换到下一个相机
    - `p`: 切换到上一个相机
- **✅畸变矫正**:
  - 通过`--calibration_file`参数加载YAML标定文件
  - 支持鱼眼相机和普通相机的畸变矫正
  - 通过`--display_undistorted`启用畸变矫正显示
- ✅**图片保存**:
  - 保存路径：`--save_path`（默认`captured_images`）
  - 保存格式：`--save_format`（默认`png`，可选`jpg`）
  - 文件名包含时间戳和相机标识
- ✅ **支持双模式标定**：普通镜头(标准针孔模型)和鱼眼镜头(专用鱼眼模型)
- ✅ **精确的FOV计算**：针对鱼眼镜头使用等距投影模型精确计算水平/垂直视场角
- ✅ **交互式角点检测**：实时显示角点检测结果，可随时中断标定过程
- ✅ **灵活的图像匹配**：支持多种命名格式的标定图像自动识别
- ✅ **详细的标定报告**：包含重投影误差、相机参数、视场角等关键指标
- ✅ **YAML格式输出**：结构化保存标定结果，便于其他程序调用
- ✅ **质量评估**：自动对标定质量进行评估并提供改进建议

## 3. 依赖安装

```
#基础依赖
pip install opencv-python numpy pyyaml
# (可选) 用于可视化和调试
pip install matplotlib
```

**最低版本要求**：

- OpenCV 4.5+
- NumPy 1.19+
- PyYAML 5.4+

## 4. 使用方法

### 4.0 使用示例

连接一个RTSP相机并保存PNG图片：

```
python camera_capture.py --cameras rtsp://192.168.1.100:8554/stream1 --save_format png
```

连接两个USB相机（设备索引0和1）并启用畸变矫正：

```
python camera_capture.py --cameras 0 1 --calibration_file calibration.yaml --display_undistorted
```

连接RTSP相机和USB相机（设备索引0）：

```
1python camera_capture.py --cameras rtsp://192.168.1.100:8554/stream1 0 --save_path ./my_images --save_format jpg
```

#### 注意事项

1. USB相机连接：脚本使用设备索引（如0,1,2）连接USB相机，而不是串口。如果您的USB相机无法通过设备索引访问，可能需要检查相机驱动或系统设置。
2. 畸变矫正：需要提供有效的YAML标定文件，格式与您提供的示例一致。
3. 鱼眼相机：当`lens_type`为`fisheye`时，脚本会使用专用的鱼眼畸变矫正函数。

### 4.1 准备标定图像

1. 打印标准棋盘格图案(推荐尺寸11x8内角点)
2. 从不同角度和距离拍摄棋盘格照片(至少15-20张)
3. 照片应覆盖相机视场的各个区域，尤其是边缘区域
4. 为获得最佳结果，确保:
   - 棋盘格完整出现在图像中
   - 有足够的光照且避免反光
   - 部分图像应使棋盘格靠近图像边缘(对鱼眼镜头尤为重要)

### 4.2 运行标定程序

```
1python camera_calibration.py \
2  --image_path ./calibration_images \
3  --width 1280 \
4  --height 720 \
5  --pattern_cols 11 \
6  --pattern_rows 8 \
7  --square_size 30.0 \
8  --lens_type fisheye \
9  --output_yaml calibration_result.yaml
```

## 5. 命令行参数详解

| 参数             | 类型  | 默认值                                         | 描述                                                 |
| ---------------- | ----- | ---------------------------------------------- | ---------------------------------------------------- |
| `--image_path`   | str   | `D:\Schen\CalibrationScript\calibrationdata_2` | 标定图片存储路径，支持相对或绝对路径                 |
| `--width`        | int   | 1280                                           | 目标图像宽度(像素)，程序会自动调整不符合此尺寸的图像 |
| `--height`       | int   | 720                                            | 目标图像高度(像素)，程序会自动调整不符合此尺寸的图像 |
| `--pattern_cols` | int   | 11                                             | 棋盘格内角点列数(宽度方向)                           |
| `--pattern_rows` | int   | 8                                              | 棋盘格内角点行数(高度方向)                           |
| `--square_size`  | float | 30.0                                           | 棋盘格方格实际尺寸(毫米)                             |
| `--lens_type`    | enum  | fisheye                                        | 镜头类型: `normal`(普通镜头)或`fisheye`(鱼眼镜头)    |
| `--output_yaml`  | str   | calibration_result_2.yaml                      | 标定结果YAML保存文件路径                             |

## 6. 输出结果说明

### 6.1 控制台输出

程序运行过程中会显示以下信息：

- 检测到的标定图像列表
- 角点检测结果(成功/失败)
- 标定过程的详细参数
- 重投影误差(RMS error)
- 相机内参矩阵和畸变系数
- 水平/垂直/对角视场角(FOV)
- 标定质量评估

### 6.2 YAML输出文件结构

```
1calibration_date: "2023-11-15 14:30:22"
2lens_type: "fisheye"
3image_resolution:
4  width: 1280
5  height: 720
6camera_matrix:
7  rows: 3
8  cols: 3
9  data: [592.4258267601276, 0.0, 639.2530105235604, 0.0, 593.1335010830682, 356.0829291965235, 0.0, 0.0, 1.0]
10distortion_coefficients:
11  count: 4
12  data: [-0.009789679197483588, 0.03567336260715891, -0.042427432224345716, 0.012132745796960275]
13field_of_view:
14  horizontal: 173.42
15  vertical: 127.65
16  diagonal: 188.31
17reprojection_error:
18  rms: 0.3562
19  unit: "pixels"
20calibration_parameters:
21  chessboard_size:
22    columns: 11
23    rows: 8
24  square_size_mm: 30.0
25  used_images_count: 23
26  total_images_count: 25
27notes:
28- "HFOV: Horizontal Field of View (水平视场角)"
29- "VFOV: Vertical Field of View (垂直视场角)"
30- "DFOV: Diagonal Field of View (对角视场角)"
31- "RMS error < 1.0 is considered good for most applications"
32- "For fisheye cameras, error < 2.0 may be acceptable due to extreme distortion"
```

## 7. 使用示例

### 7.1 普通相机标定

```
1python camera_calibration.py \
2  --image_path ./normal_camera_images \
3  --width 1920 \
4  --height 1080 \
5  --pattern_cols 9 \
6  --pattern_rows 6 \
7  --square_size 25.0 \
8  --lens_type normal \
9  --output_yaml normal_camera_calibration.yaml
```

### 7.2 鱼眼相机标定

```
1python camera_calibration.py \
2  --image_path ./fisheye_camera_images \
3  --width 1280 \
4  --height 720 \
5  --pattern_cols 11 \
6  --pattern_rows 8 \
7  --square_size 30.0 \
8  --lens_type fisheye \
9  --output_yaml fisheye_camera_calibration.yaml
```

## 8. 常见问题解决

### 8.1 无法找到标定图像

- **问题**：`错误: 在路径中未找到符合命名规则的图片`
- 解决方案
  1. 检查图像路径是否正确
  2. 确认图像是否符合 `left-*.png` 命名格式
  3. 如使用其他命名格式，程序会尝试自动匹配包含"left"的图像文件
  4. 确保图像是PNG格式，或修改代码支持其他格式

### 8.2 角点检测失败

- **问题**：多张图像中无法检测到棋盘格角点
- 解决方案
  1. 增加光照，减少反光
  2. 调整棋盘格在图像中的大小和位置
  3. 确认`pattern_cols`和`pattern_rows`参数与实际棋盘格匹配
  4. 对于鱼眼镜头，避免将棋盘格放在过度扭曲的区域

### 8.3 高重投影误差

- **问题**：标定结果的重投影误差值较高(>1.0像素)
- 解决方案
  1. 增加标定图像数量(建议20-30张)
  2. 确保图像覆盖整个视场，特别是边缘区域
  3. 检查是否选择了正确的镜头类型(普通/鱼眼)
  4. 重新打印高质量的棋盘格，确保方格尺寸精确
  5. 对于鱼眼镜头，误差<2.0通常可接受

### 8.4 鱼眼标定失败

- **问题**：`鱼眼标定失败: ...`，回退到普通相机模型
- 解决方案
  1. 检查是否真的是鱼眼镜头
  2. 增加更多覆盖边缘区域的标定图像
  3. 尝试使用更大尺寸的棋盘格
  4. 在极端畸变区域，考虑使用专业标定板替代棋盘格

## 9. 技术细节与算法说明

### 9.1 视场角(FOV)计算

- 普通镜头：使用标准针孔相机模型计算

  ```
  1HFOV = 2 * arctan(width / (2 * fx))
  ```

- 鱼眼镜头：使用等距投影模型精确计算

  ```
  theta_left = asin(cx / fx)
  theta_right = asin((W - 1 - cx) / fx)
  HFOV = (theta_left + theta_right) * 180 / π
  ```

### 9.2 鱼眼镜头标定

本工具使用OpenCV的`cv::fisheye`模块进行鱼眼镜头标定，该模型考虑了鱼眼镜头的非线性投影特性，使用4个畸变参数(k1,k2,k3,k4)而非普通相机的5个参数。

### 9.3 标定质量评估标准

- **优秀**：RMS误差 < 0.5像素
- **良好**：0.5 ≤ RMS误差 < 1.0像素
- **一般**：1.0 ≤ RMS误差 < 2.0像素
- **较差**：RMS误差 ≥ 2.0像素