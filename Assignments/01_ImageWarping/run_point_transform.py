import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """

    ### FILL: 基于MLS or RBF 实现 image warping
    h, w = image.shape[:2]
    warped_image = np.zeros_like(image)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    for idx, v in enumerate(coords):
        x, y = v
        weights = 1.0 / (np.power(np.linalg.norm(source_pts - v, axis=1), 2 * alpha) + eps)
        p_star = np.dot(weights, source_pts) / np.sum(weights)
        q_star = np.dot(weights, target_pts) / np.sum(weights)

        p_hat = source_pts - p_star
        q_hat = target_pts - q_star
        p_hat_bot = np.column_stack((-p_hat[:, 1], p_hat[:, 0]))
        q_hat_bot = np.column_stack((-q_hat[:, 1], q_hat[:, 0]))

        part1 = np.dot(weights, np.sum(q_hat * p_hat, axis=1))
        part2 = np.dot(weights, np.sum(q_hat * p_hat_bot, axis=1))
        mu = np.sqrt(part1 ** 2 + part2 ** 2)

        m11 = np.dot(weights, np.sum(p_hat * q_hat, axis=1))
        m12 = np.dot(weights, np.sum(-p_hat * q_hat_bot, axis=1))
        m21 = np.dot(weights, np.sum(-p_hat_bot * q_hat, axis=1))
        m22 = np.dot(weights, np.sum(p_hat_bot * q_hat_bot, axis=1))

        M = np.array([[m11, m12], [m21, m22]]) / mu

        original_p = (v - q_star) @ np.linalg.inv(M) + p_star
        original_x, original_y = original_p

        if 0 <= original_x < w - 1 and 0 <= original_y < h - 1:
            x1, y1 = int(original_x), int(original_y)
            x2, y2 = x1 + 1, y1 + 1

            dx = original_x - x1
            dy = original_y - y1

            top_left = image[y1, x1]
            top_right = image[y1, x2]
            bottom_left = image[y2, x1]
            bottom_right = image[y2, x2]

            warped_image[y, x] = (
                    (1 - dx) * (1 - dy) * top_left +
                    dx * (1 - dy) * top_right +
                    (1 - dx) * dy * bottom_left +
                    dx * dy * bottom_right
            )

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
