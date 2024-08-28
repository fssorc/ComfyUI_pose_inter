import torch

import comfy.utils

import cv2
import numpy as np
import folder_paths
import os


connect_color = [
    [  0,   0, 255],
    [255,   0,   0],
    [255, 170,   0],
    [255, 255,   0],
    [255,  85,   0],
    [170, 255,   0],
    [ 85, 255,   0],
    [  0, 255,   0],

    [  0, 255,  85],
    [  0, 255, 170],
    [  0, 255, 255],
    [  0, 170, 255],
    [  0,  85, 255],
    [ 85,   0, 255],

    [170,   0, 255],
    [255,   0, 255],
    [255,   0, 170],
    [255,   0,  85]
]

for i, (R, G, B) in enumerate(connect_color):
    connect_color[i] = [B,G,R]

# 骨架连接的关节对
skeleton = [
    [0, 1],   [1, 2],  [2, 3],   [3, 4],
    [1, 5],   [5, 6],  [6, 7],   [1, 8],
    [8, 9],   [9, 10], [1, 11],  [11, 12],
    [12, 13], [14, 0], [14, 16], [15, 0],
    [15, 17]
]

def gen_skeleton(pose_keypoints_2d, canvas_width, canvas_height,landmarkType):
    # 加载背景图片或创建一个空白画布
    image = None  #cv2.imread('background.jpg')  # 使用实际的背景图片路径
    if image is None:
        image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    if landmarkType == "DWPose":
        canvas_height=1
        canvas_width=1

    tri_tuples = [pose_keypoints_2d[i:i + 3] for i in range(0, len(pose_keypoints_2d), 3)]
    # 绘制骨架
    for i, (a, b) in enumerate(skeleton):
        a_x, a_y, a_z = tri_tuples[a]
        a_x, a_y    = ( a_x * canvas_width, a_y * canvas_height )
        b_x, b_y, b_z = tri_tuples[b]
        b_x, b_y    = ( b_x * canvas_width, b_y * canvas_height )
        
        if a_z != 0 and b_z != 0:
            cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), connect_color[i] + [0], 4)

    # 绘制关键点
    for i, (x, y, z) in enumerate(tri_tuples):
        if z!=0:
            cv2.circle(image, (int(x * canvas_width), int(y * canvas_height)), 6, connect_color[i], -1)
            #cv2.putText(image, str(i), (int(x * canvas_width), int(y * canvas_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def transform_keypoints(keypoints_1, keypoints_2, frames):
    tri_tuples_1 = [keypoints_1[i:i + 3] for i in range(0, len(keypoints_1), 3)]
    tri_tuples_2 = [keypoints_2[i:i + 3] for i in range(0, len(keypoints_2), 3)]

    keypoints_array = [keypoints_1]
    for j in range(1, frames):
        kp = []
        for i in range(len(tri_tuples_1)):
            x1, y1, z1 = tri_tuples_1[i]
            x2, y2, z2 = tri_tuples_2[i]

            if z1 == 0 and  z2 == 0:
                new_x, new_y, new_z = (0.0, 0.0, 0.0)
            elif z1 == 0:
                new_x, new_y, new_z = (x2, y2, z2)
            elif z2 == 0:
                new_x, new_y, new_z = (x1, y1, z1)
            else:
                new_x, new_y, new_z  = ( x1 + (x2-x1) * j/frames, y1 + (y2-y1) * j/frames , 1.0)

            kp.append( new_x)
            kp.append( new_y)
            kp.append( new_z)
        keypoints_array.append(kp)
    #keypoints_array.append(keypoints_2)

    return keypoints_array
            

           
class Pose_Inter:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_from": ("POSE_KEYPOINT", ),
                "pose_to": ("POSE_KEYPOINT", ),
                "interpolate_frames": ("INT", {"default": 10, "min": 2, "max": 100, "step": 1}),
                "landmarkType": (["OpenPose","DWPose"], ),
                
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "Pose Interpolation"

    def run(self,pose_from,pose_to,interpolate_frames,landmarkType):

        openpose_dict_2 = pose_from[0]
        openpose_dict = pose_to[0]

        keypoints_array = transform_keypoints(
            openpose_dict_2["people"][0]["pose_keypoints_2d"],
            openpose_dict["people"][0]["pose_keypoints_2d"],
            interpolate_frames
        )
        output=[]
        #print("image shape")
        #print(image.shape)
        for i, keypoints in enumerate(keypoints_array):

            # 显示图像
            image = gen_skeleton(
                keypoints, 
                openpose_dict_2["canvas_width"], 
                openpose_dict_2["canvas_height"],
                landmarkType
            )
            image = torch.from_numpy(image.astype(np.float32) / 255.0)#.unsqueeze(0)    
            output.append(image)

        tensor_stacked = torch.stack(output)
        #print("shape of tensor_stacked")
        #print(tensor_stacked.shape)
        return (tensor_stacked,)
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Pose_Inter": Pose_Inter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "Pose_Inter": "Pose Interpolation"
}
