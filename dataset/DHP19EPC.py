# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 19:48
# @Author  : Jiaan Chen
# The SA-SimDR part is
# modified from repository of "https://github.com/leeyegy/SimDR"


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from os.path import join
import torch
import cv2
import copy
from time import time
from .sample import random_sample_point
from .rasterized import RasEventCloud


def adjust_target_weight(joint, target_weight, tmp_size, sx=346, sy=260):
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= sx or ul[1] >= sy or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        target_weight = 0

    return target_weight


def generate_sa_simdr(joints, joints_vis, sigma=8, sx=346, sy=260, num_joints=13):
    """
    joints:  [num_joints, 3]
    joints_vis: [num_joints, 3]

    return => target, target_weight(1: visible, 0: invisible)
    """

    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target_x = np.zeros((num_joints, int(sx)), dtype=np.float32)
    target_y = np.zeros((num_joints, int(sy)), dtype=np.float32)

    tmp_size = sigma * 3

    frame_size = np.array([sx, sy])
    frame_resize = np.array([sx, sy])
    feat_stride = frame_size / frame_resize

    for joint_id in range(num_joints):
        target_weight[joint_id] = \
            adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
        if target_weight[joint_id] == 0:
            continue

        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        x = np.arange(0, int(sx), 1, np.float32)
        y = np.arange(0, int(sy), 1, np.float32)

        v = target_weight[joint_id]
        if v > 0.5:
            target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
            target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))

            # norm to [0,1]
            target_x[joint_id] = (target_x[joint_id] - target_x[joint_id].min()) / (
                    target_x[joint_id].max() - target_x[joint_id].min())
            target_y[joint_id] = (target_y[joint_id] - target_y[joint_id].min()) / (
                    target_y[joint_id].max() - target_y[joint_id].min())

    return target_x, target_y, target_weight


def generate_label(u, v, mask, sigma=8, sx=346, sy=260, num_joints=13):
    joints_3d = np.zeros((num_joints, 3), dtype=float)
    joints_3d_vis = np.zeros((num_joints, 3), dtype=float)
    joints_3d[:, 0] = u
    joints_3d[:, 1] = v
    joints_3d_vis[:, 0] = mask
    joints_3d_vis[:, 1] = mask

    gt_x, gt_y, gt_joints_weight = generate_sa_simdr(joints_3d, joints_3d_vis, sigma, sx=sx, sy=sy, num_joints=num_joints)

    return gt_x, gt_y, gt_joints_weight


class DHP19EPC(Dataset):
    def __init__(self, args, root_data_dir=None, root_label_dir=None,
                 root_3Dlabel_dir=None, root_dict_dir=None, min_EventNum=1024, Test3D=False):
        self.root_data_dir = root_data_dir
        self.root_label_dir = root_label_dir
        self.root_3Dlabel_dir = root_3Dlabel_dir
        self.Test3D = Test3D
        self.sample_point_num = args.num_points
        self.label = args.label
        self.sx = args.sensor_sizeW
        self.sy = args.sensor_sizeH
        self.sigma = getattr(args, 'sigma', 8)
        self.num_joints = getattr(args, 'num_joints', 13)

        if not self.Test3D:
            if self.label == 'mean':
                # filter with dict
                self.Point_Num_Dict = np.load(root_dict_dir, allow_pickle=True).item()
                self.min_EventNum = min_EventNum
                self.dvs_frames = []
                for k, v in self.Point_Num_Dict.items():
                    if ((k[-5] == '2') or (k[-5] == '3')) and (v >= self.min_EventNum):
                        self.dvs_frames.append(os.path.join(self.root_data_dir, k))
            elif self.label == 'last':
                # self.dvs_frames0 = sorted(glob.glob(os.path.join(root_data_dir, "*cam0*.npy")))
                # self.dvs_frames1 = sorted(glob.glob(os.path.join(root_data_dir, "*cam1*.npy")))
                self.dvs_frames2 = sorted(glob.glob(os.path.join(root_data_dir, "*cam2*.npy")))
                # self.dvs_frames3 = sorted(glob.glob(os.path.join(root_data_dir, "*cam3*.npy")))

                # use cam2 only for monocular pseudo-labels
                self.dvs_frames = self.dvs_frames2 # + self.dvs_frames3

        else:
            if self.label == 'mean':
                # filter with dict
                self.Point_Num_Dict = np.load(root_dict_dir, allow_pickle=True).item()
                self.min_EventNum = min_EventNum
                self.dvs_frames = []
                for k, v in self.Point_Num_Dict.items():
                    if (k[-5] == '2') and (v >= self.min_EventNum):
                        temp = k[:-5] + '3' + k[-4:]
                        if self.Point_Num_Dict[temp] >= self.min_EventNum:
                            self.dvs_frames.append(os.path.join(self.root_data_dir, k))
            else:
                print('Last Label no Test3D')

    def __getitem__(self, item):

        if not self.Test3D:
            pointcloud, xlabel, ylabel, wlabel = self.load_sample(item)

            return pointcloud, xlabel, ylabel, wlabel
        else:
            pointcloud, xlabel, ylabel, wlabel, label3D = self.load_sample(item)

            return pointcloud, xlabel, ylabel, wlabel, label3D

    def __len__(self):
        return len(self.dvs_frames)

    def load_sample(self, item=None):

        file_dir = copy.deepcopy(self.dvs_frames[item])

        if not self.Test3D:
            # Training Mode (Single View)
            pcdata = np.load(file_dir)  # [N, 4]: [x, y, t, p]
            if self.label == 'last':
                pcdata = pcdata[:, [0, 2, 1, 3]]  # last label[N, 4]: [x, t, y, p] -> [x, y, t, p] (assuming swap needed)
            
            # --- Auto-Denormalize Check ---
            # If coordinates are [0, 1], scale them up to sensor size
            if pcdata.shape[0] > 0:
                if np.max(pcdata[:, 0]) <= 1.05 and np.max(pcdata[:, 1]) <= 1.05:
                    if item == 0:
                         print(f"Detected normalized data for {item}. Scaling by ({self.sx}, {self.sy}).")
                    pcdata[:, 0] *= self.sx
                    pcdata[:, 1] *= self.sy
                
                # If polarity is [-1, 1], shift to [0, 1]
                if np.min(pcdata[:, 3]) < -0.1:
                    if item == 0:
                         print("Detected [-1, 1] polarity. shifting to [0, 1].")
                    pcdata[:, 3] = (pcdata[:, 3] + 1) / 2

            # Preprocess
            data = self.RasEventCloud_preprocess(pcdata)

            # Load Label
            # data_name: frame_000000_cam2
            data_name = os.path.basename(file_dir).split('.')[0]
            
            # Extract frame index
            # data_name format "frame_XXXXXX_camX"
            parts = data_name.split('_')
            frame_idx = parts[1] # "000000"
            
            # Construct label name: frame_000000_label.npy
            label_name = f"frame_{frame_idx}_label.npy"
            # Look in the same directory as the data file
            label_file = os.path.join(os.path.dirname(file_dir), label_name)
            
            # Check if file exists
            if os.path.exists(label_file):
                pclabel = np.load(label_file)
                u, v, mask = pclabel[:].astype(float)
            else:
                # Missing label, return zeros (masked out by weight)
                u = np.zeros(13)
                v = np.zeros(13)
                mask = np.zeros(13)

            # Print debug for missing files occasionally? No, too spammy.
            
            # Joint Filtering for Arm Pose
            
            # Joint Filtering for Arm Pose
            if self.num_joints == 6:
                target_indices = [1, 2, 3, 4, 5, 6]
                u = u[target_indices]
                v = v[target_indices]
                mask = mask[target_indices]

            x, y, weight = generate_label(u, v, mask, self.sigma, sx=self.sx, sy=self.sy, num_joints=self.num_joints)
            return data, x, y, weight

        else:
            # Test3D Mode (Multi View)
            data_name = os.path.basename(file_dir).split('.')[0]
            frame_name = data_name[0:data_name.find('_cam')]

            data_cam2_file = os.path.join(self.root_data_dir, frame_name + '_cam2.npy')
            data_cam3_file = os.path.join(self.root_data_dir, frame_name + '_cam3.npy')
            label_cam2_file = os.path.join(self.root_label_dir, frame_name + '_cam2_label.npy')
            label_cam3_file = os.path.join(self.root_label_dir, frame_name + '_cam3_label.npy')

            label3D_file = os.path.join(self.root_3Dlabel_dir, frame_name + '_3Dlabel.npy')

            pcdata2 = np.load(data_cam2_file)  # [N, 4]
            pcdata3 = np.load(data_cam3_file)  # [N, 4]

            pclabel2 = np.load(label_cam2_file)
            pclabel3 = np.load(label_cam3_file)

            data2 = self.RasEventCloud_preprocess(pcdata2)
            data3 = self.RasEventCloud_preprocess(pcdata3)

            u2, v2, mask2 = pclabel2[:].astype(float)
            x2, y2, weight2 = generate_label(u2, v2, mask2, self.sigma, sx=self.sx, sy=self.sy, num_joints=13) 

            u3, v3, mask3 = pclabel3[:].astype(float)
            x3, y3, weight3 = generate_label(u3, v3, mask3, self.sigma, sx=self.sx, sy=self.sy, num_joints=13)

            label3D = np.load(label3D_file)

            return [data2, data3], [x2, x3], [y2, y3], [weight2, weight3], label3D

    def RasEventCloud_preprocess(self, data):

        if data.size == 0:

            data = np.zeros((1, 5))
            num_sample = self.sample_point_num

            if num_sample != 0:
                data_sample, select_index = random_sample_point(data, num_sample)
                data = data_sample

            return data

        data = data[:, 0:4]
        EventCloudDHP = RasEventCloud(input_size=(4, self.sy, self.sx))

        data = EventCloudDHP.convert(data).numpy()[:, 1:]  # [x, y, t_avg, p_acc, event_cnt]

        if data.shape[0] == 0:
            return np.zeros((self.sample_point_num, 5))

        num_sample = self.sample_point_num
        if num_sample != 0:
            data_sample, select_index = random_sample_point(data, num_sample)
            data = data_sample  # [num_sample, C]

        return data
