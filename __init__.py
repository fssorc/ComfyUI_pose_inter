import torch

import comfy.utils

import cv2
import numpy as np
import folder_paths
import os
import math

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
bonesPoints = [
[1,0],# neck
[0,14   ],#right eye
[0,15   ],#left eye
[14,16  ],# right ear
[15,17  ],# left ear
[1,2    ],#right shoulder
[1,5    ],#left shoulder
[2,3    ],#right elbow
[5,6    ],#left elbow
[3,4    ],#right wrist
[6,7    ],#left wrist
[1,8    ],#right hip
[1,11   ],#left hip
[8,9    ],#right knee
[11,12  ],# left knee
[9,10   ],#right ankle
[12,13  ] # left ankle
]

class HumanPart:
    def __init__(self, index):
        self.index = index
        self.abs_x = 0.0 #绝对坐标
        self.abs_y = 0.0 #  绝对坐标
        self.length = 0.0 #长度
        self.score=1.0   #置信度
        self.abs_angle=0.0 #绝对角度
        self.ref_angle=0.0 #相对角度
        self.children = [] #子物体
        self.ref_scale=1.0
        self.abs_scale=1.0
        self.parent = None #父物体

    def SetParent(self, parent):
        self.parent = parent
        parent.children.append(self)

    #根据相对角度和长度计算绝对坐标
    def Update(self):

        # 如果有父节点，则更新绝对位置和角度
        if self.parent is not None:
            if self.score > 0.0:
                # 更新绝对缩放
                self.abs_scale = self.parent.abs_scale * self.ref_scale
                # 更新绝对角度
                self.abs_angle = self.parent.abs_angle + self.ref_angle
            # 更新绝对x坐标
            self.abs_x = self.parent.abs_x + self.length * math.cos(self.abs_angle)* self.abs_scale
            # 更新绝对y坐标
            self.abs_y = self.parent.abs_y + self.length * math.sin(self.abs_angle)* self.abs_scale            
        else:            
            # 如果没有父节点，则绝对角度和缩放等于参考角度和缩放
            self.abs_angle = self.ref_angle
            self.abs_scale = self.ref_scale
        #self.score=1.0
        # 遍历子节点，递归调用Update方法
        for child in self.children:
            child.Update()

    #根据绝对位置，计算角度、相对位置和相对角度
    def CalculateRef(self):
           
        # 如果有父节点，则更新相对位置和角度
        if self.parent is not None:
            if self.score > 0.0:
                self.length = math.sqrt((self.abs_x - self.parent.abs_x)**2 + (self.abs_y - self.parent.abs_y)**2)
                self.abs_angle =math.atan2(self.abs_y - self.parent.abs_y, self.abs_x - self.parent.abs_x)
                self.ref_angle = self.abs_angle - self.parent.abs_angle 
                self.ref_scale = self.abs_scale / self.parent.abs_scale
                #如果父节点score为0，而本节点score为1，那么这个节点就只有abs_x,abs_y是可靠的，其他都是不可靠的
                #除非使用IK算法，否则无法计算可靠的角度和长度。超出了这个插件编写的本意。留着以后优化吧。
                #openpose的线条，在两个端点都可靠的情况下才会绘制。因此本插件即使不计算这个骨骼，也不影响绘制。
                #本插件还有猜测不可靠节点的位置的功能，不放在这个函数里处理，是因为这个函数是递归调用的，放在这里的话，猜测的位置会不断被覆盖。
                #猜测功能用在pose interpolation里，因为设计本意里，pose interpolation的输入来自两张不同的图片，里面的可靠节点数量不一致，所以需要猜测不可靠节点的位置。
                
        else:
            self.length = 0.0
            self.ref_angle = 0.0
            self.abs_angle= 0.0

        # 遍历子节点，递归调用Update方法    
        for child in self.children:
            child.CalculateRef()



class HumanDoll:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        self.parts = [None] * (len(bonesPoints) + 1)
        self.keyPoints=[]

        self.RootPart = HumanPart(1) # 根部 固定长度为0，对应keypoints[1]的位置
        self.abs_x = canvas_width / 2
        self.abs_y = canvas_height / 3
        self.RootPart.abs_x = canvas_width / 2
        self.RootPart.abs_y = canvas_height / 3
        self.RootPart.ref_angle = 0.0
        self.RootPart.score = 1.0
        self.parts[1] = self.RootPart

        for (p,c) in bonesPoints: # p为父节点，c为子节点  
            child = HumanPart(c)
            child.SetParent(self.parts[p])
            self.parts[c] = child

    def UpdateByKeypoints(self, keypoints):
        self.keyPoints = keypoints
        for i, keypoint in enumerate(keypoints):
            if keypoint is not None:
                self.parts[i].abs_x = keypoint[0]
                self.parts[i].abs_y = keypoint[1]
                self.parts[i].score = keypoint[2]
            else:
                self.parts[i].score = 0.0
        self.RootPart.CalculateRef()
        
    def UpdateByParts(self):
        self.RootPart.Update()
        for i in range(len(self.keyPoints)):            
            self.keyPoints[i] = (self.parts[i].abs_x, self.parts[i].abs_y, self.parts[i].score)

    def Draw(self, width, height,draw_all=False):
        # 在这里实现绘制函数
        return keyPoints_2_image(self.keyPoints, width, height,draw_all)
    
    def guessMissingParts(self):
        # 在这里实现猜测缺失部分的功能。用于pose interpolation
        doll_T_Pose = HumanDoll(self.canvas_width,self.canvas_height)
        keyPoints = gen_t_pose_keypoints(self.canvas_width,self.canvas_height)
        doll_T_Pose.UpdateByKeypoints(keyPoints)
        # 定义各部位缺失时的参考顺序数组
        # 格式为：(目标部位索引, [参考部位列表按优先级排序])
        reference_order = [
            (0, [2,5,8,11]),    # 补充头部（部位0）时参考的顺序
            (2, [5,8,11,0]),    # 补充右肩（部位2）时参考的顺序
            (5, [2,8,11,0]),    # 补充左肩（部位5）时参考的顺序
            (8, [11,2,5,0]),    # 补充右髋（部位8）时参考的顺序
            (11, [8,2,5,0]),    # 补充左髋（部位11）时参考的顺序
            (3, [6,2,8,11,0]),  # 补充右上臂（部位3）时参考的顺序
            (6, [3,2,11,8,0]),  # 补充左上臂（部位6）时参考的顺序
            (4, [3,7,2,5,8,11,0]),  # 补充右前臂（部位4）时参考的顺序
            (7, [4,6,5,3,2,8,11,0]),  # 补充左前臂（部位7）时参考的顺序
            (9, [12,8,11,2,5,0]),  # 补充右大腿（部位9）时参考的顺序
            (12, [9,11,8,5,2,0]),  # 补充左大腿（部位12）时参考的顺序
            (10, [13,9,12,8,11,2,5,0]),  # 补充右小腿（部位10）时参考的顺序
            (13, [10,12,9,11,8,2,5,0]),  # 补充左小腿（部位13）时参考的顺序

            #以上都是以父节点长度参考缺失节点长度，而角度则从t-pose中复制
            
            #但是脸上的部位，需要另外的算法。实际运行中出现过耳朵可靠而眼睛不可靠的情况，因此需要另外的算法
            # (14, [15,0]),  # 补充右眼睛（部位14）时参考的顺序
            # (15, [14,0]),  # 补充左眼睛（部位15）时参考的顺序
            # (16, [17,14,15,0]),  # 补充右耳朵（部位16）时参考的顺序
            # (17, [16,15,14,0]),  # 补充左耳朵（部位17）时参考的顺序
        
            # 其他部位的参考规则可以继续添加
            # ...
        ]

        # 遍历所有定义好的参考规则
        for target_part, reference_parts in reference_order:
            # 跳过1号部位（根据需求描述1号部位不会缺失）
            if target_part == 1:
                continue
            
            # 如果目标部位缺失（分数为0）
            if self.parts[target_part].score == 0.0:

                # 标记为已尝试修复
                self.parts[target_part].score = 1.0
                # 复制T姿态下的参考角度
                self.parts[target_part].ref_angle = doll_T_Pose.parts[target_part].ref_angle
                
                # 遍历参考部位列表，按优先级尝试补充数据
                for ref_part in reference_parts:
                    if self.parts[ref_part].score > 0.0:
                        # 使用参考部位数据补充目标部位长度
                        # 比例因子：目标部位在T姿态下的长度 / 参考部位在T姿态下的长度
                        ratio = doll_T_Pose.parts[target_part].length / doll_T_Pose.parts[ref_part].length
                        #if target_part == 16 or target_part == 17:
                        
                        self.parts[target_part].length = self.parts[ref_part].length * ratio
                        #print(f"[{target_part}]-[{ref_part}]-ratio:{ratio}- self.parts[ref_part].length:{self.parts[ref_part].length} - result_length:{self.parts[target_part].length}")
                        break  # 找到第一个有效参考后退出循环
                else:
                    # 如果所有参考部位都无效，重置分数
                    self.parts[target_part].score = 0.0

            #现在处理右眼。如果右眼缺失，则根据左耳朵和鼻子的位置计算
        if self.parts[14].score == 0.0:
            self.parts[14].score=1.0
            if self.parts[16].score > 0.0 and self.parts[0].score > 0.0:
                self.parts[14].abs_x = (self.parts[16].abs_x + self.parts[0].abs_x)/2
                self.parts[14].abs_y = (self.parts[16].abs_y + self.parts[0].abs_y)/2
                self.parts[14].CalculateRef()
            elif self.parts[0].score > 0.0:
                ratio = doll_T_Pose.parts[14].length / doll_T_Pose.parts[0].length
                self.parts[14].length = self.parts[0].length * ratio
            else:
                self.parts[14].score = 0.0

        #现在处理左眼。如果左眼缺失，则根据右耳朵和鼻子的位置计算
        if self.parts[15].score == 0.0:
            self.parts[15].score=1.0
            if self.parts[17].score > 0.0 and self.parts[0].score > 0.0:
                self.parts[15].abs_x = (self.parts[17].abs_x + self.parts[0].abs_x)/2
                self.parts[15].abs_y = (self.parts[17].abs_y + self.parts[0].abs_y)/2
                self.parts[15].CalculateRef()
            elif self.parts[0].score > 0.0:
                ratio = doll_T_Pose.parts[15].length / doll_T_Pose.parts[0].length
                self.parts[15].length = self.parts[0].length * ratio
            else:
                self.parts[15].score = 0.0

        reference_order_part2 = [

            (16, [17,14,15,0]),  # 补充右耳朵（部位16）时参考的顺序
            (17, [16,15,14,0]),  # 补充左耳朵（部位17）时参考的顺序

        ]

        for target_part, reference_parts in reference_order_part2:
            # 跳过1号部位（根据需求描述1号部位不会缺失）
            if target_part == 1:
                continue
            
            # 如果目标部位缺失（分数为0）
            if self.parts[target_part].score == 0.0:

                # 标记为已尝试修复
                self.parts[target_part].score = 1.0
                # 复制T姿态下的参考角度
                self.parts[target_part].ref_angle = doll_T_Pose.parts[target_part].ref_angle
                
                # 遍历参考部位列表，按优先级尝试补充数据
                for ref_part in reference_parts:
                    if self.parts[ref_part].score > 0.0:
                        # 使用参考部位数据补充目标部位长度
                        # 比例因子：目标部位在T姿态下的长度 / 参考部位在T姿态下的长度
                        ratio = doll_T_Pose.parts[target_part].length / doll_T_Pose.parts[ref_part].length
                        #if target_part == 16 or target_part == 17:
                        
                        self.parts[target_part].length = self.parts[ref_part].length * ratio
                        #print(f"[{target_part}]-[{ref_part}]-ratio:{ratio}- self.parts[ref_part].length:{self.parts[ref_part].length} - result_length:{self.parts[target_part].length}")
                        break  # 找到第一个有效参考后退出循环
                else:
                    # 如果所有参考部位都无效，重置分数
                    self.parts[target_part].score = 0.0

            #现在处理右眼。如果右眼缺失，则根据左耳朵和鼻子的位置计算
        
        print("修复完成")


    
            



        



def gen_t_pose_keypoints(canvas_width, canvas_height):
    # 初始化关键点数组
    keyPoints = [None] * (len(bonesPoints) + 1)

    # 定义起始点（1号关键点，颈部）的位置
    start_x = canvas_width / 2
    start_y = canvas_height / 4
    keyPoints[1] = (start_x, start_y, 1.0)

    # 定义骨骼长度
    neck_length = canvas_height / 10
    shoulder_length = canvas_width / 8
    arm_length = canvas_width / 6
    hip_width = shoulder_length
    hip_height = canvas_height / 4
    leg_length = canvas_height / 5
    head_radius = canvas_height / 15  # 头部半径
    eye_distance = canvas_width / 10  # 眼睛间距
    ear_distance = canvas_width / 10  # 耳朵间距

    # 计算0号关键点（头部中心）的位置
    head_center_x = start_x
    head_center_y = start_y - neck_length
    keyPoints[0] = (head_center_x, head_center_y, 1.0)

    # 计算右肩（2号关键点）的位置
    right_shoulder_x = start_x - shoulder_length
    right_shoulder_y = start_y
    keyPoints[2] = (right_shoulder_x, right_shoulder_y, 1.0)

    # 计算左肩（5号关键点）的位置
    left_shoulder_x = start_x + shoulder_length
    left_shoulder_y = start_y
    keyPoints[5] = (left_shoulder_x, left_shoulder_y, 1.0)

    # 计算右肘（3号关键点）的位置
    right_elbow_x = right_shoulder_x - arm_length
    right_elbow_y = right_shoulder_y
    keyPoints[3] = (right_elbow_x, right_elbow_y, 1.0)

    # 计算左肘（6号关键点）的位置
    left_elbow_x = left_shoulder_x + arm_length
    left_elbow_y = left_shoulder_y
    keyPoints[6] = (left_elbow_x, left_elbow_y, 1.0)

    # 计算右手腕（4号关键点）的位置
    right_wrist_x = right_elbow_x - arm_length
    right_wrist_y = right_elbow_y
    keyPoints[4] = (right_wrist_x, right_wrist_y, 1.0)

    # 计算左手腕（7号关键点）的位置
    left_wrist_x = left_elbow_x+ arm_length
    left_wrist_y = left_elbow_y
    keyPoints[7] = (left_wrist_x, left_wrist_y, 1.0)

    # 计算右髋（8号关键点）的位置
    right_hip_x = start_x - hip_width
    right_hip_y = start_y + hip_height
    keyPoints[8] = (right_hip_x, right_hip_y, 1.0)

    # 计算左髋（11号关键点）的位置
    left_hip_x = start_x + hip_width
    left_hip_y = start_y + hip_height
    keyPoints[11] = (left_hip_x, left_hip_y, 1.0)

    # 计算右膝（9号关键点）的位置
    right_knee_x = right_hip_x
    right_knee_y = right_hip_y + leg_length
    keyPoints[9] = (right_knee_x, right_knee_y, 1.0)

    # 计算左膝（12号关键点）的位置
    left_knee_x = left_hip_x
    left_knee_y = left_hip_y + leg_length
    keyPoints[12] = (left_knee_x, left_knee_y, 1.0)

    # 计算右脚踝（10号关键点）的位置
    right_ankle_x = right_knee_x
    right_ankle_y = right_knee_y + leg_length 
    keyPoints[10] = (right_ankle_x, right_ankle_y, 1.0)

    # 计算左脚踝（13号关键点）的位置
    left_ankle_x = left_knee_x
    left_ankle_y = left_knee_y + leg_length 
    keyPoints[13] = (left_ankle_x, left_ankle_y, 1.0)

    # 计算右眼（14号关键点）的位置
    right_eye_x = head_center_x - eye_distance / 2
    right_eye_y = head_center_y- eye_distance / 2
    keyPoints[14] = (right_eye_x, right_eye_y, 1.0)

    # 计算左眼（15号关键点）的位置
    left_eye_x = head_center_x + eye_distance / 2
    left_eye_y = head_center_y- eye_distance / 2
    keyPoints[15] = (left_eye_x, left_eye_y, 1.0)

    # 计算右耳（16号关键点）的位置
    right_ear_x = head_center_x - ear_distance
    right_ear_y = head_center_y
    keyPoints[16] = (right_ear_x, right_ear_y, 1.0)

    # 计算左耳（17号关键点）的位置
    left_ear_x = head_center_x + ear_distance
    left_ear_y = head_center_y
    keyPoints[17] = (left_ear_x, left_ear_y, 1.0)

    return keyPoints

def poseKeypoints2D_2_keypoints(pose_keypoints_2d, canvas_width, canvas_height,landmarkType):
    keyPoints = [pose_keypoints_2d[i:i + 3] for i in range(0, len(pose_keypoints_2d), 3)]
    if landmarkType == "DWPose":
        canvas_height=1
        canvas_width=1
    for i in range(len(keyPoints)):
        x, y, z = keyPoints[i]
        keyPoints[i] = (x * canvas_width, y * canvas_height, z)
    return keyPoints

def keypoints_2_bones(keyPoints):
    bones=[]
    for i, (a, b) in enumerate(bonesPoints):
        a_x, a_y, a_z = keyPoints[a]
        b_x, b_y, b_z = keyPoints[b]
        # 现在 (a_x,a_y) 是骨骼起点的坐标，(b_x,b_y) 是骨骼终点的坐标
        # 然后计算骨骼的长度和角度，连同坐标一起生成一个骨骼对象，放入bones数组
        # 如果a_z=0或者b_z=0，则忽略这个骨骼，放入一个空对象
        if a_z == 0 or b_z == 0:
            bone = {
            'start': (a_x, a_y),
            'end': (b_x, b_y),
            'length': -1,
            'angle': 0
            }
            bones.append(bone)   
            continue

        # 计算骨骼的长度
        bone_length = math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)

        # 计算骨骼的角度
        bone_angle = math.atan2(b_y - a_y, b_x - a_x)

        # 生成骨骼对象
        bone = {
            'start': (a_x, a_y),
            'end': (b_x, b_y),
            'length': bone_length,
            'angle': bone_angle                
        }
        bones.append(bone)
    return bones

def bones_2_keypoints(bones):
    keyPoints = [None] * 18
    # 从1号关键点开始
    start_bone = bones[0]
    start_x, start_y = start_bone['start']
    neck_length = start_bone['length']
    neck_angle = start_bone['angle']
    keyPoints[1] = (start_x ,start_y,1.0)
    # 计算新的0号关键点的位置
    new_x = start_x + neck_length * math.cos(neck_angle)
    new_y = start_y + neck_length * math.sin(neck_angle)
    keyPoints[0] = (new_x ,new_y,1.0)
    
    # 依次计算后续关键点的位置
    for i, (a, b) in enumerate(bonesPoints):
        if i == 0:
            continue  # 跳过第一个骨骼，因为它已经处理过了
        # 获取前一个关键点的位置
        prev_x,prev_y,score = keyPoints[a]
        
        # 获取当前骨骼的信息
        bone = bones[i]
        bone_length = bone['length']
        bone_angle = bone['angle']
        if bone_length == -1:       
            keyPoints[b]=(0,0,0)              
            continue
        # 计算新的关键点位置
        new_x = prev_x + bone_length * math.cos(bone_angle)
        new_y = prev_y + bone_length * math.sin(bone_angle)

        # 将新的关键点位置存入数组
        keyPoints[b]=(new_x ,new_y,1.0)  
    return keyPoints

def keyPoints_2_image(keyPoints, canvas_width, canvas_height ,draw_all=False):
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        # 绘制骨架
    for i, (a, b) in enumerate(skeleton):
        a_x, a_y, a_z = keyPoints[a]
        b_x, b_y, b_z = keyPoints[b]
        if draw_all:
            if a_z != 0 and b_z != 0:
                cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), connect_color[i] + [0], 4)
            else:
                cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), (255,255,255), 4)
        else:
            # 检查是否在画面内（假设a_z和b_z为0表示不在画面内）
            a_in_canvas = a_z != 0 and 0 <= a_x < canvas_width and 0 <= a_y < canvas_height
            b_in_canvas = b_z != 0 and 0 <= b_x < canvas_width and 0 <= b_y < canvas_height
            if a_in_canvas and b_in_canvas:
                cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), connect_color[i] + [0], 4)
    # 绘制关键点
    for i, (x, y, z) in enumerate(keyPoints):
        if z!=0:
            try:
                cv2.circle(image, (int(x), int(y)), int(canvas_width/100), connect_color[i], -1)
            except:
                pass
    #         cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)        
    return image


def draw_skeleton(pose_keypoints_2d, canvas_width, canvas_height,landmarkType):
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
            image = draw_skeleton(
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


           
class PoseModify:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", ),
                "landmarkType": (["OpenPose","DWPose"], ),
                "globalScale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "head": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "neck": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "neck": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "torsoWidth": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "torsoHeight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "upperArmLength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "lowerArmLength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "upperLegLength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),
                "lowerLegLength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0,"step":0.1}),

            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "Pose Interpolation"

    def run(self,pose_keypoint,landmarkType,globalScale,head,neck,torsoWidth,torsoHeight,upperArmLength,lowerArmLength,upperLegLength,lowerLegLength):
        # if head==1.0 & neck==1.0 & torsoWidth==1.0 & torsoHeight==1.0 & upperArmLength==1.0 & lowerArmLength==1.0 & upperLegLength==1.0 & lowerLegLength==1.0:
        #     lengthFactors = [1.0] * 18
        # else:
        lengthFactors = [neck,1.0,
                            torsoWidth,
                            upperArmLength,
                            lowerArmLength,
                            torsoWidth,
                            upperArmLength,
                            lowerArmLength,
                            torsoHeight,
                            upperLegLength,
                            lowerLegLength,
                            torsoHeight,
                            upperLegLength,
                            lowerLegLength,
                            head,
                            head,
                            head,
                            head]
                             
        openpose_dict_2 = pose_keypoint[0]
        width = openpose_dict_2["canvas_width"]
        height = openpose_dict_2["canvas_height"]
        output=[]
        openpose_dict_2 = pose_keypoint[0]
        width = openpose_dict_2["canvas_width"]
        height = openpose_dict_2["canvas_height"]
        keyPoints = gen_t_pose_keypoints(width,height)
        doll = HumanDoll(width,height)
        doll.UpdateByKeypoints(keyPoints)

        for openpose_dict_2 in pose_keypoint:
            human1Keypoints = poseKeypoints2D_2_keypoints(
                openpose_dict_2["people"][0]["pose_keypoints_2d"],
                width,
                height,
                landmarkType
                )
            
            doll.UpdateByKeypoints(human1Keypoints)
            doll.RootPart.ref_scale = globalScale
            for i in range(len(doll.parts)):
                if doll.parts[i].score > 0.0:
                    doll.parts[i].length *= lengthFactors[i]
            
            doll.UpdateByParts()
            image=doll.Draw(width,height)

            image = torch.from_numpy(image.astype(np.float32) / 255.0)#.unsqueeze(0)    
            output.append(image)

        tensor_stacked = torch.stack(output)
        #print("shape of tensor_stacked")
        #print(tensor_stacked.shape)
        return (tensor_stacked,)
    
class GenTPose:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 480, "min": 2, "max": 1024, "step": 1}),
                "height": ("INT", {"default": 640, "min": 2, "max": 1024, "step": 1}),
                
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "Pose Interpolation"

    def run(self,width,height):

        keyPoints = gen_t_pose_keypoints(width,height)
        doll = HumanDoll(width, height)
        doll.UpdateByKeypoints(keyPoints)
        doll.UpdateByParts()
        
        image=doll.Draw(width, height)
        output=[]
        image = torch.from_numpy(image.astype(np.float32) / 255.0)#.unsqueeze(0)    
        output.append(image)

        tensor_stacked = torch.stack(output)
        #print("shape of tensor_stacked")
        #print(tensor_stacked.shape)
        return (tensor_stacked,)    
        
class Pose_Inter_V2:
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
                "draw_all": ("BOOLEAN", {"default": False}),
                
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "Pose Interpolation"

    def run(self,pose_from,pose_to,interpolate_frames,landmarkType,draw_all):
        
        output=[]
        openpose_dict = pose_from[0]
        openpose_dict_2 = pose_to[0]
        


        width = openpose_dict_2["canvas_width"]
        height = openpose_dict_2["canvas_height"]
        dollFrom = HumanDoll(width,height)
        fromKeys = poseKeypoints2D_2_keypoints(openpose_dict["people"][0]["pose_keypoints_2d"],width,height,landmarkType)
        dollFrom.UpdateByKeypoints(fromKeys)

        dollTo = HumanDoll(width,height)
        toKeys = poseKeypoints2D_2_keypoints(openpose_dict_2["people"][0]["pose_keypoints_2d"],width,height,landmarkType)
        dollTo.UpdateByKeypoints(toKeys)

        dollFrom.guessMissingParts()
        dollTo.guessMissingParts()


        image=dollFrom.Draw(width,height,draw_all)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        output.append(image)
        position_x_arr = np.full(interpolate_frames+2, -1.0, dtype=np.float32)
        position_y_arr = np.full(interpolate_frames+2,-1.0, dtype=np.float32)
        position_x_arr[0] = dollFrom.RootPart.abs_x
        position_y_arr[0] = dollFrom.RootPart.abs_y
        position_x_arr[interpolate_frames+1] = dollTo.RootPart.abs_x
        position_y_arr[interpolate_frames+1] = dollTo.RootPart.abs_y

        for i in range(1, interpolate_frames+1):
            position_x_arr[i] = position_x_arr[0] + (position_x_arr[interpolate_frames+1] - position_x_arr[0]) * (i - 1) / interpolate_frames
            position_y_arr[i] = position_y_arr[0] + (position_y_arr[interpolate_frames+1] - position_y_arr[0]) * (i - 1) / interpolate_frames
        #18个骨骼，每个骨骼2个数组，每个数组有interpolate_frames+2个元素表示骨骼每一帧的长度和角度
        curveDatas = np.full((18, 2, interpolate_frames+2), -1.0, dtype=np.float32)

        for i in range(len(dollFrom.parts)):
            if dollFrom.parts[i].score > 0.0:
                curveDatas[i][0][0] = dollFrom.parts[i].length
                curveDatas[i][1][0] = dollFrom.parts[i].ref_angle
            dollFrom.parts[i].score=1.0

        for i in range(len(dollTo.parts)):
            if dollTo.parts[i].score > 0.0:
                curveDatas[i][0][interpolate_frames+1] = dollTo.parts[i].length
                curveDatas[i][1][interpolate_frames+1] = dollTo.parts[i].ref_angle
            dollTo.parts[i].score=1.0

        for i in range(18):
            j=0
            #骨骼长度，线性插值
            for k in range(1, interpolate_frames+1):
                curveDatas[i][j][k] = curveDatas[i][j][0] + (curveDatas[i][j][interpolate_frames+1] - curveDatas[i][j][0]) * (k) / (interpolate_frames+1)


        for i in range(18):
            j=1
            a = curveDatas[i][j][0]
            b = curveDatas[i][j][interpolate_frames+1]
            # 将 b 调整到与 a 变化量最小的等效弧度值
            b_adjusted = b - 2 * np.pi * np.round((b - a) / (2 * np.pi))
            curveDatas[i][j][interpolate_frames+1] = b_adjusted

            for k in range(1, interpolate_frames+1):
                curveDatas[i][j][k] = a + (b_adjusted - a) * (k) / (interpolate_frames+1)
            curveDatas[i][j] = np.mod(curveDatas[i][j], 2 * np.pi)

        for i in range(1,interpolate_frames+1):
            dollFrom.RootPart.abs_x = position_x_arr[i]
            dollFrom.RootPart.abs_y = position_y_arr[i]
            for k in range(18):
                dollFrom.parts[k].length = curveDatas[k][0][i]
                dollFrom.parts[k].ref_angle = curveDatas[k][1][i]
            dollFrom.UpdateByParts()
            image=dollFrom.Draw(width,height,draw_all)
            image = torch.from_numpy(image.astype(np.float32) / 255.0)#.unsqueeze(0)    
            output.append(image)

        image=dollTo.Draw(width,height,draw_all)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)#.unsqueeze(0)    
        output.append(image)
        
        tensor_stacked = torch.stack(output)
        
        return (tensor_stacked,)
            
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Pose_Inter": Pose_Inter,
    "Pose_Inter_V2": Pose_Inter_V2,
    "PoseModify": PoseModify,
    "GenTPose":GenTPose
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
      "Pose_Inter": "Pose Interpolation",
      "Pose_Inter_V2": "Pose Interpolation_V2",
      "PoseModify": "Pose Modify by factor",
      "GenTPose": "GenTPose"
}
