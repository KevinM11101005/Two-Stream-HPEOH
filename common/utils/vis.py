import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
from main.config import cfg

def draw_skeletons(numpy_image, keypoints):
    cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    if cfg.trainset == 'DEX_YCB':
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
            ]
    elif cfg.trainset == 'HO3D':
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 17),
            (0, 4), (4, 5), (5, 6), (6, 18),
            (0, 7), (7, 8), (8, 9), (9, 20),
            (0, 10), (10, 11), (11, 12), (12, 19),
            (0, 13), (13, 14), (14, 15), (15, 16)
        ]
    for point_1, point_2 in connections:
        x1, y1 = keypoints[0][point_1][0], keypoints[0][point_1][1]
        x2, y2 = keypoints[0][point_2][0], keypoints[0][point_2][1]
        cv2.circle(cv_image, (int(x1), int(y1)), 4, (0, 255, 0), -1)  # 在关键点处画一个半径为 3 的绿色圆
        cv2.circle(cv_image, (int(x2), int(y2)), 4, (0, 255, 0), -1)  # 在关键点处画一个半径为 3 的绿色圆
        cv2.line(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 画一条连接两个关键点的线段
    # 显示或保存图像
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('Tensor to CV Image', cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv_image

def draw_bbox(img_path):
    # 定义局部变量
    rect = None
    start_point = None
    end_point = None
    drawing = False

    # 读取图像
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # 鼠标回调函数
    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_point, end_point, rect, drawing

        x = max(0, min(x, width - 1))  # 限制 x 坐标在图像宽度范围内
        y = max(0, min(y, height - 1))  # 限制 y 坐标在图像高度范围内

        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False
            rect = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]), 
                    abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]))

    # 读取图像
    image = cv2.imread(img_path)

    # 创建窗口并绑定鼠标回调函数
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    # 在图像上绘制矩形
    while True:
        draw_img = image.copy()
        if start_point and end_point:
            cv2.rectangle(draw_img, start_point, end_point, (0, 0, 255), 2)

        cv2.imshow('image', draw_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:  # Esc 键
            break
        elif key == ord('r'):
            start_point = None
            end_point = None
            rect = None
    cv2.destroyAllWindows()
    return rect

def ux_hon_result(orig_img, img, pred_skeleton_map, gt_skeleton_map):
    img_uint8 = cv2.resize(orig_img.astype(np.uint8), (cfg.input_img_shape[0], cfg.input_img_shape[1]))
    restored_img = np.expand_dims(pred_skeleton_map*255, axis=-1)
    final_image = np.expand_dims(gt_skeleton_map*255, axis=-1)
    restored_img = cv2.cvtColor(restored_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    final_image = cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    rgb_img_uint8 = cv2.cvtColor(img_uint8.astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    ori_imgs = np.hstack([rgb_img_uint8, rgb_img])
    re_imgs = np.hstack([restored_img, final_image])
    cat_imgs = np.vstack([ori_imgs, re_imgs])
    return cat_imgs.astype(np.uint8)

def ux_hon_result_final(out, bb2img_trans, orig_img, img, cat_imgs):
    original_img_height, original_img_width = orig_img.shape[:2]
    img = img.astype(np.uint8)
    if cfg.simcc:
        # get hand keypoints and keypoint_scores
        keypoints, keypoint_scores = out['keypoints'], out['keypoint_scores']
        keypoints = np.expand_dims(keypoints, axis=0)
        keypoints_restored = np.dot(bb2img_trans, np.concatenate((keypoints[0], np.ones((keypoints[0].shape[0], 1))), axis=1).transpose(1, 0))
        keypoints_restored = keypoints_restored[:2, :].transpose(1, 0)
        final_image = draw_skeletons(orig_img, np.expand_dims(keypoints_restored,axis=0))
        bbox_image = draw_skeletons(img, keypoints)
    else:
        keypoints = out['joints_coord_img'].cpu().numpy()
        keypoints = np.expand_dims(keypoints, axis=0)
        keypoints[:,:,0] *= cfg.input_img_shape[1]
        keypoints[:,:,1] *= cfg.input_img_shape[0]
        keypoints_restored = np.dot(bb2img_trans, np.concatenate((keypoints[0], np.ones((keypoints[0].shape[0], 1))), axis=1).transpose(1, 0))
        keypoints_restored = keypoints_restored[:2, :].transpose(1, 0)
        final_image = draw_skeletons(orig_img, np.expand_dims(keypoints_restored,axis=0))
        bbox_image = draw_skeletons(img, keypoints)
        
    bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
    final_image = cv2.resize(final_image, (cfg.input_img_shape[0], cfg.input_img_shape[1]))
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    rgb_imgs = np.hstack([bbox_image, final_image])
    cat_imgs = np.vstack([cat_imgs, rgb_imgs])
    return cat_imgs.astype(np.uint8)

def overlay_bbox_image(origin_image, bbox_image, bbox_coords):
    # 获取边界框坐标
    x_min, y_min, width, height = bbox_coords
    y_max, x_max, _ = origin_image.shape
    if x_min < 0:
        x_range = x_min+width-1
        x_min = 0
    else:
        x_range = x_min+width-1
    if y_min < 0:
        y_range = y_min+height-1
        y_min = 0
    else:
        y_range = y_min+height-1
    # 将 bbox_image 覆盖回 origin_image 的相应部分
    origin_image[y_min:min(y_range, y_max), x_min:min(x_range, x_max)] = bbox_image[y_min:min(y_range, y_max), x_min:min(x_range, x_max)]
    # print(f'y_min:{y_min}, x_min:{x_min}')
    # print(f'y_max:{min(y_range, y_max)}, x_max:{min(x_range, x_max)}')
    return origin_image

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    #plt.show()
    #cv2.waitKey(0)

    plt.savefig(filename)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def render_mesh(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask*0.5 + img #* (1-valid_mask)
    return img