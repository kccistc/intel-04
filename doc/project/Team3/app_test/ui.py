import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
import os
from tkinter import messagebox , Toplevel
from PIL import Image, ImageTk, ImageSequence
import cv2
import mediapipe as mp

###### pose estimation

import glob
import collections
import time
from pathlib import Path
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
import openvino as ov
import ipywidgets as widgets
######

window_width = 1600
window_height = 1200
pad = 50
current_dir = os.path.dirname(__file__)

model_path = os.path.join(current_dir,"resource","model")
pose_model = os.path.join(model_path,"human-pose-estimation-0001.xml")
md_model= os.path.join(model_path,"model.keras")

image_path = os.path.join(current_dir,"resource","image")
squart_path = os.path.join(image_path,"squart.png")
pig_image = os.path.join(image_path,"pig.png")
background_image = os.path.join(image_path,"walk.gif")


######model compile
md_core = ov.Core()

md_device =widgets.Dropdown(
    options = md_core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False
)
md_model = md_core.read_model(md_model)
md_compiled_model = md_core.compile_model(model=md_model, device_name=md_device.value, config={"PERFORMANCE_HINT": "LATENCY"})
md_output_layers = md_compiled_model.output("Mconv7_stage2_L2")


######pose set up
core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)
model = core.read_model(pose_model)
compiled_model = core.compile_model(model=model, device_name=device.value, config={"PERFORMANCE_HINT": "LATENCY"})

pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")

input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs
height, width = list(input_layer.shape)[2:]

# code from https://github.com/openvinotoolkit/open_model_zoo/blob/9296a3712069e688fe64ea02367466122c8e8a3b/demos/common/python/models/open_pose.py#L135
class OpenPoseDecoder:
    BODY_PARTS_KPT_IDS = (
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (1, 11),
        (11, 12),
        (12, 13),
        (1, 0),
        (0, 14),
        (14, 16),
        (0, 15),
        (15, 17),
        (2, 16),
        (5, 17),
    )
    BODY_PARTS_PAF_IDS = (
        12,
        20,
        14,
        16,
        22,
        24,
        0,
        2,
        4,
        6,
        8,
        10,
        28,
        30,
        34,
        32,
        36,
        18,
        26,
    )

    def __init__(
        self,
        num_joints=18,
        skeleton=BODY_PARTS_KPT_IDS,
        paf_indices=BODY_PARTS_PAF_IDS,
        max_points=100,
        score_threshold=0.1,
        min_paf_alignment_score=0.05,
        delta=0.5,
    ):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta

        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))

        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self.convert_to_coco_format(pose_entries, keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        return poses, scores

    def extract_points(self, heatmaps, nms_heatmaps):
        batch_size, channels_num, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"
        assert channels_num >= self.num_joints

        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for k in range(self.num_joints):
            # Filter low-score points.
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            # Apply quarter offset to improve localization accuracy.
            x, y = self.refine(heatmaps[0, k], x, y)
            np.clip(x, 0, w - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
            # Pack resulting points.
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(keypoints)
        return all_keypoints

    def top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        # Get positions with top scores.
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points :]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        # Keep top scores sorted.
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    @staticmethod
    def refine(heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(x > 0, x < w - 1), np.logical_and(y > 0, y < h - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    @staticmethod
    def is_disjoint(pose_a, pose_b):
        pose_a = pose_a[:-2]
        pose_b = pose_b[:-2]
        return np.all(np.logical_or.reduce((pose_a == pose_b, pose_a < 0, pose_b < 0)))

    def update_poses(
        self,
        kpt_a_id,
        kpt_b_id,
        all_keypoints,
        connections,
        pose_entries,
        pose_entry_size,
    ):
        for connection in connections:
            pose_a_idx = -1
            pose_b_idx = -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
            if pose_a_idx < 0 and pose_b_idx < 0:
                # Create new pose entry.
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                # Merge two poses are disjoint merge them, otherwise ignore connection.
                pose_a = pose_entries[pose_a_idx]
                pose_b = pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                # Adjust score of a pose.
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                # Add a new limb into pose.
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                # Add a new limb into pose.
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        # From all retrieved connections that share starting/ending keypoints leave only the top-scoring ones.
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_kpt_a = set()
        has_kpt_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        # For every limb.
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            # Sample points along every candidate limb vector.
            steps = 1 / (self.points_per_limb - 1) * vec_raw
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Compute affinity score between candidate limb vectors and part affinity field.
            part_pafs = pafs[0, :, :, paf_channel : paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            # Get a list of limbs according to the obtained affinity score.
            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            # Suppress incompatible connections.
            a_idx, b_idx, affinity_scores = self.connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(
                zip(
                    kpts_a[a_idx, 3].astype(np.int32),
                    kpts_b[b_idx, 3].astype(np.int32),
                    affinity_scores,
                )
            )
            if len(connections) == 0:
                continue

            # Update poses with new connections.
            pose_entries = self.update_poses(
                kpt_a_id,
                kpt_b_id,
                all_keypoints,
                connections,
                pose_entries,
                pose_entry_size,
            )

        # Remove poses with not enough points.
        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(-1, pose_entry_size)
        pose_entries = pose_entries[pose_entries[:, -1] >= 3]
        return pose_entries, all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose[-2]
            for keypoint_id, target_id in zip(pose[:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))  # -1 for 'neck'
        return np.asarray(coco_keypoints), np.asarray(scores)
    
decoder = OpenPoseDecoder()

# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = (
        img.shape[1] / output_shape[3].get_length(),
        img.shape[0] / output_shape[2].get_length(),
    )
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores

colors = (
    (255, 0, 0),
    (255, 0, 255),
    (170, 0, 255),
    (255, 0, 85),
    (255, 0, 170),
    (85, 255, 0),
    (255, 170, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 85),
    (170, 255, 0),
    (0, 85, 255),
    (0, 255, 170),
    (0, 0, 255),
    (0, 255, 255),
    (85, 0, 255),
    (0, 170, 255),
)

default_skeleton = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(
                    img_limbs,
                    tuple(points[i]),
                    tuple(points[j]),
                    color=colors[j],
                    thickness=4,
                )
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img

class angle_calculater:
    def __init__(self):
        self.IMG1_INDEX = 0
        self.IMG2_INDEX = 1
        self.IMG3_INDEX = 2
        self.IMG4_INDEX = 3
        self.IMG5_INDEX = 4
        self.IMG6_INDEX = 5

        self.LEFT_NECK_INDEX = 0
        self.RIGHT_NECK_INDEX = 1
        self.LEFT_SHOULDER_INDEX = 2
        self.LEFT_INSIDE_SHOULDER_INDEX = 3
        self.LEFT_ELBOW_INDEX = 4
        self.LEFT_ARMPIT_INDEX = 5
        self.RIGHT_INSIDE_SHOULDER_INDEX = 6
        self.RIGHT_SHOULDER_INDEX = 7
        self.RIGHT_ARMPIT_INDEX = 8
        self.RIGHT_ELBOW_INDEX = 9
        self.LEFT_PELVIS_INDEX = 10
        self.LEFT_HIP_INDEX = 11
        self.RIGHT_PELVIS_INDEX = 12
        self.RIGHT_HIP_INDEX = 13
        self.LEFT_LEG_INDEX = 14
        self.RIGHT_LEG_INDEX = 15
        self.LEFT_KNEE_INDEX = 16
        self.RIGHT_KNEE_INDEX = 17
        self.all_angles = []

        left_neck_angle = None
        right_neck_angle = None
        left_shoulder_angle = None
        left_inside_shoulder_angle = None
        left_elbow_angle = None
        left_armpit_angle = None
        right_inside_shoulder_angle = None
        right_shoulder_angle = None
        right_armpit_angle = None
        right_elbow_angle = None
        left_pelvis_angle = None
        left_hip_angle = None
        right_pelvis_angle = None
        right_hip_angle = None
        left_leg_angle = None
        right_leg_angle = None
        left_knee_angle = None
        right_knee_angle = None

    def calculate_angles(self, poses):
        for pose in poses:
            joint = np.zeros((18, 2))
            for j, lm in enumerate(pose):
                joint[j] = [lm[0], lm[1]]

            # Compute angles between joints
            v1 = joint[[1, 1, 1, 5, 5, 6, 6, 7, 8, 12, 11, 13, 12, 14], :]
            v2 = joint[[2, 5, 6, 7, 11, 12, 8, 9, 10, 11, 13, 15, 14, 16], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arccos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 0, 1, 1, 3, 3, 2, 2, 6, 6, 4, 4, 5, 5, 9, 9, 10, 12], :],
                                        v[[1, 2, 3, 4, 7, 4, 5, 6, 5, 8, 9, 10, 9, 12, 10, 12, 11, 13], :]))
            angle = np.degrees(angle)
            self.all_angles.append(angle)
            #print("All angles:", self.all_angles)
        return self.all_angles

img_angle = angle_calculater()

def image_init(img_path):
    frame = cv2.imread(img_path)

    input_img = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_AREA)
    # Create a batch of images (size = 1).
    input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

    # Measure processing time.
    start_time = time.time()
    # Get results.
    results = compiled_model([input_img])
    stop_time = time.time()

    pafs = results[pafs_output_key]
    heatmaps = results[heatmaps_output_key]
    # Get poses from network results.
    poses, scores = process_results(frame, pafs, heatmaps)

    all_angle = img_angle.calculate_angles(poses)
    left_neck = all_angle[img_angle.IMG1_INDEX][img_angle.LEFT_NECK_INDEX]
    return left_neck
  
#def degree():
######


#####

def show_home():
    hide_all_frames()
    home_frame.pack(fill="both", expand=True)
    update_nav_style(home_link)

def show_menu1():
    if not username:
        return
    hide_all_frames()
    menu1_frame.pack(fill="both", expand=True)
    update_nav_style(menu1_link)
    video_capture.start()
    open_new_window(squart_path)

def show_menu2():
    if not username:
        return
    hide_all_frames()
    menu2_frame.pack(fill="both", expand=True)
    update_nav_style(menu2_link)

def show_menu3():
    if not username:
        return
    hide_all_frames()
    menu3_frame.pack(fill="both", expand=True)
    update_nav_style(menu3_link)

def show_about():
    if not username:
        return
    hide_all_frames()
    about_frame.pack(fill="both", expand=True)
    update_nav_style(about_link)

def hide_all_frames():
    home_frame.pack_forget()
    menu1_frame.pack_forget()
    menu2_frame.pack_forget()
    menu3_frame.pack_forget()
    about_frame.pack_forget()
    video_capture.stop()
    disable_widgets(home_frame)
    disable_widgets(menu1_frame)
    disable_widgets(menu2_frame)
    disable_widgets(menu3_frame)
    disable_widgets(about_frame)
    
    global current_image_window
    if current_image_window:
        current_image_window.destroy()
        current_image_window = None
    

def disable_widgets(frame):
    for child in frame.winfo_children():
        child.configure(state='disabled')

def confirm_username(event=None):
    global username
    username = username_entry.get().strip()
    if username:
        username_label.grid_forget()
        username_entry.grid_forget()
        confirm_button.grid_forget()
        update_username_display()
        center_frame.pack_forget()
    else:
        messagebox.showwarning("Warning", "Please enter a value!") 

def update_username_display():
    global username_display
    username_display = ttk.Label(collapse, text=f"Welcome,  {username}!", style='Primary.TLabel', font=('Roboto', 12, 'normal'))
    username_display.pack(side="right", padx=10)

def update_nav_style(label):
    for link in [home_link, menu1_link, menu2_link, menu3_link, about_link]:
        link.config(font=('Roboto', 12, 'normal'))
    
    label.config(font=('Roboto', 12, 'bold'))

def open_new_window(image_path):
    global current_image_window
    
    if current_image_window:
        current_image_window.destroy()
    
    new_window = Toplevel(window)
    new_window.title("가이드")
    # 이미지 로드 및 라벨에 설정
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(new_window, image=photo)
    label.image = photo  # 이 줄을 추가하여 이미지가 가비지 컬렉션되지 않도록 방지
    label.pack(padx=20, pady=20)
    
    # 현재 이미지 창 업데이트
    current_image_window = new_window

def create_back_button(frame,command):
    button = tk.Button(frame, text="뒤로", command=command)
    button.pack(pady=10)

class VideoCapture:
    def __init__(self, canvas, x, y, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cap = None
        # self.cap2 = cv2.VideoCapture(2)
        self.image_id = None
        self.data = None
        self.is_running = False
        
    def start(self):
        self.cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠) 사용
        self.is_running = True
        self.update()

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()

    def update(self):
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        # ret, frame2 = self.cap2.read()
        frame = frame[:,1*self.width//4 :3*self.width//4]
        # frame2 = frame2[:,1*self.width//4 :3*self.width//4]

        self.data=[]

        if ret:
            self.current_frame = frame.copy()  # 현재 프레임을 변수에 저장
            data = []
            ##### pose estimation
            scale = 1280 / max(frame.shape)
            # scale2 = 1280 / max(frame2.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                # frame2 = cv2.resize(frame2, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_AREA)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # input_img2 = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)
            input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
            # input_img2 = input_img2.transpose((2, 0, 1))[np.newaxis, ...]
            results = compiled_model([input_img])
            # results2 = compiled_model([input_img2])
            pafs = results[pafs_output_key]
            # pafs2 = results2[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # heatmaps2 = results2[heatmaps_output_key]
            poses, scores = process_results(frame, pafs, heatmaps)
            # poses2, scores2 = process_results(frame2, pafs2, heatmaps2)
            frame = draw_poses(frame, poses, 0.1)
            # frame2 = draw_poses(frame2, poses2, 0.1)
            ######
            for pose in poses:
                joint = np.zeros((18, 2))
                for j, lm in enumerate(pose):
                    joint[j] = [lm[0], lm[1]]

                # Compute angles between joints
                v1 = joint[[1, 1, 1, 5, 5, 6, 6, 7, 8, 12, 11, 13, 12, 14], :]
                v2 = joint[[2, 5, 6, 7, 11, 12, 8, 9, 10, 11, 13, 15, 14, 16], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arccos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 0, 1, 1, 3, 3, 2, 2, 6, 6, 4, 4, 5, 5, 9, 9, 10, 12], :],
                                            v[[1, 2, 3, 4, 7, 4, 5, 6, 5, 8, 9, 10, 9, 12, 10, 12, 11, 13], :]))
                angle = np.degrees(angle)
                print(f"{angle[0]},{left_neck} ")


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            frame = cv2.resize(frame, (self.width, self.height))  # 프레임 크기 조정
            # frame2 = cv2.resize(frame2, (self.width, self.height))  # 프레임 크기 조정
            img = Image.fromarray(frame)
            # img2 = Image.fromarray(frame2)
            imgtk = ImageTk.PhotoImage(image=img)
            # imgtk2 = ImageTk.PhotoImage(image=img2)
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(self.x, self.y, anchor='nw', image=imgtk)
            # self.image_id = self.canvas.create_image(window_width //2, self.y, anchor='nw', image=imgtk2)
            self.canvas.image = imgtk
            # self.canvas.image2 = imgtk2

        self.canvas.after(10, self.update)  # 10ms마다 업데이트

left_neck=image_init(squart_path)

# tkinter 윈도우 생성
window = tk.Tk()
window.title("Project Logo")
window.geometry(f"{window_width}x{window_height}")

style = Style(theme='minty')

##### 시작 모드 프레임 생성
start_frame = ttk.Frame(window)
start_frame.pack(fill="x")

# 레이블 생성
title = ttk.Label(start_frame, text="Project_Logo", style='primary.TLabel', font=('Roboto', 18))
title.grid(row=0, column=0, padx=10)

title_toggler = ttk.Button(start_frame, text="☰", style='TButton')
title_toggler.grid(row=0, column=1)

collapse = ttk.Frame(start_frame)
collapse.grid(row=0, column=2, padx=10)

nav = ttk.LabelFrame(collapse, style='secondary.TLabelFrame')
nav.pack(side="left", padx=10, pady=10)

# 메뉴 링크
home_link = ttk.Label(nav, text="Home", style='secondary.TLabel', cursor="hand2")
home_link.grid(row=0, column=0, padx=10)
home_link.bind("<Button-1>", lambda event: show_home())

menu1_link = ttk.Label(nav, text="Menu1", style='secondary.TLabel', cursor="hand2")
menu1_link.grid(row=0, column=1, padx=10)
menu1_link.bind("<Button-1>", lambda event: show_menu1())

menu2_link = ttk.Label(nav, text="Menu2", style='secondary.TLabel', cursor="hand2")
menu2_link.grid(row=0, column=2, padx=10)
menu2_link.bind("<Button-1>", lambda event: show_menu2())

menu3_link = ttk.Label(nav, text="Menu3", style='secondary.TLabel', cursor="hand2")
menu3_link.grid(row=0, column=3, padx=10)
menu3_link.bind("<Button-1>", lambda event: show_menu3())

about_link = ttk.Label(nav, text="About", style='secondary.TLabel', cursor="hand2")
about_link.grid(row=0, column=4, padx=10)
about_link.bind("<Button-1>", lambda event: show_about())

# 입력 필드 생성
center_frame = ttk.Frame(window)
center_frame.pack(pady=150)

username_label = ttk.Label(center_frame, text="Enter your username:")
username_label.grid(row=0, column=0, pady=10)

username_entry = ttk.Entry(center_frame)
username_entry.grid(row=1, column=0, pady=10)

confirm_button = ttk.Button(center_frame, text="Confirm", command=confirm_username)
confirm_button.grid(row=2, column=0, pady=10)

# Enter 키로도 입력 확인 가능하게 설정 
username_entry.bind("<Return>", confirm_username)
#####

##### 메인 메뉴의 프레임 생성
home_frame = ttk.Frame(window)

# Canvas 생성 및 배경 이미지 설정
canvas = tk.Canvas(home_frame, width=window_width, height=window_height)
canvas.pack(fill="both", expand=True)

label_home = ttk.Label(home_frame, text="Welcome to Home Page", font=('Roboto', 18))
label_home.pack(pady=20, side='bottom')

username_display = None  # 사용자명을 표시할 Label

##### menu1
menu1_frame = ttk.Frame(window)
label_menu1 = ttk.Label(menu1_frame, text="Menu1 Page", font=('Roboto', 18))
label_menu1.pack(pady=20, side='bottom')

video_canvas = tk.Canvas(menu1_frame, width=window_width, height=window_height)
video_canvas.pack(fill="both", expand=True)
video_capture = VideoCapture(video_canvas, 0,0, window_width // 2, window_height) ## 화면 오른쪽에 refernce image 추가
current_image_window = None
#####


##### menu2
menu2_frame = ttk.Frame(window)
label_menu2 = ttk.Label(menu2_frame, text="Menu2 Page", font=('Roboto', 18))
label_menu2.pack(pady=20, side='bottom')
#####


##### menu3
menu3_frame = ttk.Frame(window)
label_menu3 = ttk.Label(menu3_frame, text="Menu3 Page", font=('Roboto', 18))
label_menu3.pack(pady=20, side='bottom')
#####


##### menu4
about_frame = ttk.Frame(window)
label_about = ttk.Label(about_frame, text="About Page", font=('Roboto', 18))
label_about.pack(pady=20, side='bottom')
#####

show_home()

# tkinter 윈도우 실행
window.mainloop()