# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask

from terrain_base.terrain import Terrain
from terrain_base.config import terrain_config

from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


class HumanoidRobot(BaseTask):
    """
    人形机器人环境类
    继承自BaseTask，实现人形机器人的仿真环境
    """
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, save):
        """
        人形机器人环境初始化
        解析配置文件，创建仿真环境、地形和机器人，初始化训练用的PyTorch缓冲区
        
        Args:
            cfg (LeggedRobotCfg): 环境配置文件对象
            sim_params (gymapi.SimParams): 仿真参数
            physics_engine (gymapi.SimType): 物理引擎类型，必须是PhysX
            sim_device (string): 仿真设备 'cuda' 或 'cpu'  
            headless (bool): 如果为True则无头运行（不渲染图形）
            save (bool): 是否保存训练数据用于分析
        """
        # 保存配置参数
        self.cfg = cfg                    # 环境配置对象 (H1_2FixCfg类的实例，包含所有机器人和环境参数)
        self.sim_params = sim_params      # 仿真参数
        self.height_samples = None        # 高度图采样数据（稍后初始化）
        self.debug_viz = True            # 调试可视化开关
        self.init_done = False           # 初始化完成标志
        self.save = save                 # 数据保存标志
        
        # 解析配置文件，设置内部参数
        self._parse_cfg(self.cfg)
        
        # 调用父类初始化，创建仿真环境、地形和机器人实例
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 初始化图像处理变换（用于深度相机）
        # 将深度图像调整到指定尺寸，使用双三次插值
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        # 如果不是无头模式，设置相机视角
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
            
        # 初始化PyTorch张量缓冲区（用于存储状态、动作、奖励等）
        self._init_buffers()
        
        # 准备奖励函数（设置奖励权重和计算函数）
        self._prepare_reward_function()
    
        # 如果启用数据保存，初始化数据收集结构
        if self.save:
            self.episode_data = {
                'observations': [[] for _ in range(self.num_envs)],      # 观测数据
                'actions': [[] for _ in range(self.num_envs)],           # 动作数据
                'rewards': [[] for _ in range(self.num_envs)],           # 奖励数据
                'height_map': [[] for _ in range(self.num_envs)],        # 高度图数据
                'privileged_obs': [[] for _ in range(self.num_envs)],    # 特权观测数据
                'rigid_body_state': [[] for _ in range(self.num_envs)],  # 刚体状态数据
                'dof_state': [[] for _ in range(self.num_envs)]          # 关节状态数据
            }
            self.current_episode_buffer = {
                'observations': [[] for _ in range(self.num_envs)],      # 当前episode观测数据
                'actions': [[] for _ in range(self.num_envs)],           # 当前episode动作数据
                'rewards': [[] for _ in range(self.num_envs)],           # 当前episode奖励数据
                'height_map': [[] for _ in range(self.num_envs)],        # 当前episode高度图数据
                'privileged_obs': [[] for _ in range(self.num_envs)],    # 当前episode特权观测数据
                'rigid_body_state': [[] for _ in range(self.num_envs)],  # 当前episode刚体状态数据
                'dof_state': [[] for _ in range(self.num_envs)]          # 当前episode关节状态数据
            }
        # init data save buffer
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.time_stamp = 0

        self.total_times = 0
        self.last_times = -1
        self.success_times = 0
        self.complete_times = 0.

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def get_data_stats(self):
        """get dataset information"""
        stats = {
            'total_episodes': 0,
            'total_samples': 0,
            'avg_episode_length': 0
        }
        for env_data in self.episode_data['observations']:
            stats['total_episodes'] += len(env_data)
            for ep in env_data:
                stats['total_samples'] += ep.shape[0]
        if stats['total_episodes'] > 0:
            stats['avg_episode_length'] = stats['total_samples'] / stats['total_episodes']
        return stats

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None

        if self.save:
            for env_idx in range(self.num_envs):
                self.current_episode_buffer['observations'][env_idx].append(
                    self.obs_buf[env_idx].cpu().numpy().copy())  
                self.current_episode_buffer['actions'][env_idx].append(
                    self.actions[env_idx].cpu().numpy().copy())      
                
                self.current_episode_buffer['rewards'][env_idx].append(
                    self.rew_buf[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['height_map'][env_idx].append(
                    self.measured_heights_data[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['rigid_body_state'][env_idx].append(
                    self.rigid_body_states[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['dof_state'][env_idx].append(
                    self.dof_state[env_idx].cpu().numpy().copy())  

                if self.privileged_obs_buf is not None:
                    self.current_episode_buffer['privileged_obs'][env_idx].append(
                        self.privileged_obs_buf[env_idx].cpu().numpy().copy())      

        if(self.cfg.rewards.is_play):
            if(self.total_times > 0):
                if(self.total_times > self.last_times):
                    # print("total_times=",self.total_times)
                    # print("success_rate=",self.success_times / self.total_times)
                    # print("complete_rate=",(self.complete_times / self.total_times).cpu().numpy().copy())
                    self.last_times = self.total_times

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        """
        更新机器人的导航目标
        
        这个方法管理机器人在地形中的路径点导航：
        1. 检查是否到达当前目标点
        2. 在延迟时间后切换到下一个目标
        3. 计算相对位置和朝向角度
        4. 为奖励函数和观测提供目标信息
        """
        
        # 检查是否已经在目标点停留足够长时间，可以切换到下一个目标
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1      # 切换到下一个目标点索引
        self.reach_goal_timer[next_flag] = 0   # 重置到达目标计时器

        # 检测哪些机器人到达了当前目标点（距离小于阈值）
        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1  # 为到达目标的机器人增加计时器

        # 计算当前目标点相对于机器人的位置向量
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        # 计算下一个目标点相对于机器人的位置向量
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        # 🧭 计算目标点朝向角度（从机器人指向目标点的方向）
        # 注意：这里计算的是"导航朝向"，不是"运动命令朝向"！
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)  # 计算到目标点的距离
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)        # 归一化方向向量（避免除零）
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])  # 计算朝向目标点的偏航角

        # 🧭 计算下一个目标点的朝向角度
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)  # 计算到下个目标点的距离
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)         # 归一化方向向量
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])  # 计算朝向下个目标点的偏航角
        

    def post_physics_step(self):
        """
        物理仿真步骤后的处理
        检查终止条件，计算观测和奖励
        调用self._post_physics_step_callback()进行通用计算
        如果需要，调用self._draw_debug_vis()进行调试可视化
        """
        # 刷新仿真状态张量，获取最新的物理状态
        self.gym.refresh_actor_root_state_tensor(self.sim)    # 刷新机器人根部状态
        self.gym.refresh_net_contact_force_tensor(self.sim)   # 刷新接触力
        self.gym.refresh_rigid_body_state_tensor(self.sim)    # 刷新刚体状态
        # self.gym.refresh_force_sensor_tensor(self.sim)     # 刷新力传感器（暂未使用）

        # 更新计数器
        self.episode_length_buf += 1    # episode长度计数器递增
        self.common_step_counter += 1   # 通用步数计数器递增

        # 准备计算量：更新机器人状态信息
        self.base_quat[:] = self.root_states[:, 3:7]  # 更新基座四元数姿态
        # 将世界坐标系下的速度转换到机器人本体坐标系
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])   # 本体线速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 本体角速度
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)       # 本体坐标系下的重力方向
        # 计算基座线性加速度
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        # 从四元数计算欧拉角
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        # 检测脚部接触状态
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.  # 接触力大于2N认为接触
        self.contact_filt = torch.logical_or(contact, self.last_contacts)  # 当前接触或上一步接触都认为是接触
        self.last_contacts = contact  # 更新上一步接触状态
        
        # self._update_jump_schedule()  # 更新跳跃计划（暂未使用）
        self._update_goals()              # 更新目标点
        self._post_physics_step_callback()  # 执行后处理回调函数

        # 计算观测、奖励、重置等
        self.check_termination()  # 检查终止条件：姿态超限、高度过低、超时、完成目标
        self.compute_reward()     # 计算奖励：调用所有奖励函数并加权求和
        
        # 获取需要重置的环境ID并执行重置
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 找出需要重置的环境
        self.reset_idx(env_ids)   # 重置指定环境：清空状态、重采样命令、保存数据

        # 更新目标信息（重置后可能改变）
        self.cur_goals = self._gather_cur_goals()         # 收集当前目标点坐标
        self.next_goals = self._gather_cur_goals(future=1)  # 收集下一个目标点坐标

        # 更新深度缓冲区（如果使用视觉输入）
        self.update_depth_buffer()  # 处理深度相机图像数据

        # 计算观测（策略网络的输入）
        self.compute_observations()  # 组装本体感受观测：IMU、关节、接触、命令等

        # 更新历史状态缓冲区（用于计算动作变化率等）
        self.last_last_actions[:] = self.last_actions[:]    # 前前次动作
        self.last_actions[:] = self.actions[:]              # 前次动作
        self.last_dof_vel[:] = self.dof_vel[:]              # 前次关节速度
        self.last_torques[:] = self.torques[:]              # 前次关节力矩
        self.last_root_vel[:] = self.root_states[:, 7:13]   # 前次根部速度（线速度+角速度）
        
        # 定期更新脚部状态（每5步更新一次）
        if(self.time_stamp == 5):
            self.last_foot_action = self.rigid_body_states[:, self.feet_indices, :]  # 脚部刚体状态
            self.time_stamp = 0
        else:
            self.time_stamp = self.time_stamp + 1
        
        # 调试可视化（仅在有查看器且启用调试时）
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)      # 清除之前的调试线条
            # self._draw_height_samples()          # 绘制高度采样点（可选）
            self._draw_goals()                     # 绘制目标点
            # self._draw_feet()                    # 绘制脚部位置（可选）
            
            # 显示深度图像（如果使用深度相机）
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # 显示当前观察机器人的深度图像
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

        # 更新地形复杂度历史
        self._update_terrain_complexity_history()

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        # 策略5.2：放宽终止条件，提高容错性
        roll_cutoff = torch.abs(self.roll) > 2.0    # 原:1.5 → 新:2.0
        pitch_cutoff = torch.abs(self.pitch) > 2.0  # 原:1.5 → 新:2.0
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < 0.3  # 原:0.5 → 新:0.3

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

        self.total_times += len(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.success_times += len(reach_goal_cutoff.nonzero(as_tuple=False).flatten())
        self.complete_times += (self.cur_goal_idx[self.reset_buf.nonzero(as_tuple=False).flatten()] / self.cfg.terrain.num_goals).sum()

    def reset_idx(self, env_ids):
        """
        重置指定的环境
        调用self._reset_dofs(env_ids), self._reset_root_states(env_ids), 和 self._resample_commands(env_ids)
        可选调用self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids)
        记录episode信息并重置部分缓冲区

        Args:
            env_ids (list[int]): 需要重置的环境ID列表
        """
        if len(env_ids) == 0:  # 如果没有环境需要重置，直接返回
            return
        
        # 如果启用数据保存，处理episode数据
        if self.save:
            for env_id in env_ids:
                try:
                    # 只保存足够长的episode（超过750步）
                    if len(self.current_episode_buffer['observations'][env_id]) > 750:
                        # 将当前episode缓冲区的数据转换为numpy数组
                        episode_obs = np.stack(self.current_episode_buffer['observations'][env_id])     # 观测数据 [T,*]
                        episode_act = np.stack(self.current_episode_buffer['actions'][env_id])         # 动作数据 [T,*]
                        episode_rew = np.stack(self.current_episode_buffer['rewards'][env_id])         # 奖励数据 [T]
                        episode_hei = np.stack(self.current_episode_buffer['height_map'][env_id])      # 高度图数据 [T, 396]
                        episode_body = np.stack(self.current_episode_buffer['rigid_body_state'][env_id]) # 刚体状态 [T,13,13] 第一个是根部
                        episode_dof = np.stack(self.current_episode_buffer['dof_state'][env_id])       # 关节状态数据
                      
                        # 将episode数据存入主数据存储
                        self.episode_data['observations'][env_id].append(episode_obs)
                        self.episode_data['actions'][env_id].append(episode_act)
                        self.episode_data['rewards'][env_id].append(episode_rew)
                        self.episode_data['height_map'][env_id].append(episode_hei)
                        self.episode_data['rigid_body_state'][env_id].append(episode_body)
                        self.episode_data['dof_state'][env_id].append(episode_dof)

                        # 处理特权观测数据（如果存在）
                        if self.privileged_obs_buf is not None:
                            episode_priv = np.stack(self.current_episode_buffer['privileged_obs'][env_id]) # 特权观测 [T,*]
                            self.episode_data['privileged_obs'][env_id].append(episode_priv)
                        
                        # 清空当前episode缓冲区，为下一个episode做准备
                        self.current_episode_buffer['observations'][env_id] = []
                        self.current_episode_buffer['actions'][env_id] = []
                        self.current_episode_buffer['rewards'][env_id] = []
                        self.current_episode_buffer['height_map'][env_id] = []
                        self.current_episode_buffer['privileged_obs'][env_id] = []
                        self.current_episode_buffer['rigid_body_state'][env_id] = []
                        self.current_episode_buffer['dof_state'][env_id] = []
                        
                        print(f"Env {env_id} have saved {episode_obs.shape[0]} step data")
                except Exception as e:
                    print(f"An error occured when saving env {env_id}: {str(e)}")
        
        # 更新课程学习
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)  # 更新地形难度
        # 避免在每一步都更新命令课程，因为最大命令对所有环境都是通用的
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)   # 更新命令难度

        # 重置机器人状态
        self._reset_dofs(env_ids)           # 重置关节状态（位置、速度）
        self._reset_root_states(env_ids)    # 重置根部状态（位置、姿态、速度）
        self._resample_commands(env_ids)    # 重新采样运动命令
        
        # 执行一步仿真以应用重置
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 重置缓冲区数据
        self.last_last_actions[env_ids] = 0.      # 前前次动作
        self.last_actions[env_ids] = 0.           # 前次动作
        self.last_foot_action[env_ids] = 0.       # 前次脚部动作
        self.last_dof_vel[env_ids] = 0.           # 前次关节速度
        self.last_torques[env_ids] = 0.           # 前次力矩
        self.last_root_vel[:] = 0.                # 前次根部速度
        self.feet_air_time[env_ids] = 0.          # 脚部离地时间
        self.reset_buf[env_ids] = 1               # 重置标志
        self.obs_history_buf[env_ids, :, :] = 0.  # 观测历史缓冲区 TODO: 考虑不使用0初始化
        self.contact_buf[env_ids, :, :] = 0.      # 接触缓冲区
        self.action_history_buf[env_ids, :, :] = 0.  # 动作历史缓冲区
        self.cur_goal_idx[env_ids] = 0            # 当前目标索引
        self.reach_goal_timer[env_ids] = 0        # 到达目标计时器

        # 重置地形复杂度缓冲区
        self.terrain_complexity_ptr[env_ids] = 0
        self.terrain_complexity_history[env_ids] = 0

        # 填充额外信息（用于日志记录）
        self.extras["episode"] = {}
        # 计算并记录各项奖励的平均值
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.  # 重置奖励累计
        self.episode_length_buf[env_ids] = 0      # 重置episode长度

        # 记录额外的课程学习信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())  # 平均地形难度
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]       # 最大前向速度命令
        
        # 向算法发送超时信息
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ 
        计算奖励函数
        调用所有非零权重的奖励函数（在 self._prepare_reward_function() 中处理）
        将每个奖励项累加到 episode 总和和总奖励中
        """
        
        # ========== 步骤1：初始化奖励缓冲区 ==========
        # self.rew_buf: 形状为 [num_envs] 的张量，存储每个环境的总奖励
        # 每个时间步开始时重置为0，然后累加各项奖励reward_names
        self.rew_buf[:] = 0.
        
        # ========== 步骤2：计算并累加所有奖励项 ==========
        # 遍历所有已注册的奖励函数（权重非零的奖励项）
        # self.reward_functions: 奖励函数列表，例如 [_reward_lin_vel_z, _reward_orientation, ...]
        # self.reward_names: 奖励名称列表，例如 ['lin_vel_z', 'orientation', ...]
        # self.reward_scales: 奖励权重字典，例如 {'lin_vel_z': -2.0, 'orientation': -1.0, ...}
        for i in range(len(self.reward_functions)):
            # 获取当前奖励项的名称
            name = self.reward_names[i]
            
            # 调用奖励函数并乘以权重
            # reward_functions[i](): 调用奖励函数，返回形状为 [num_envs] 的张量
            # reward_scales[name]: 该奖励项的权重（可正可负）
            # 例如：_reward_lin_vel_z() 返回 [0.1, 0.05, 0.2, ...]，权重为 -2.0
            #       则 rew = [0.1, 0.05, 0.2, ...] * (-2.0) = [-0.2, -0.1, -0.4, ...]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            
            # 累加到总奖励缓冲区
            # 每个环境的总奖励 = 所有奖励项的加权和
            self.rew_buf += rew
            
            # 累加到 episode 奖励统计（用于日志记录和分析）
            # 排除特殊的统计奖励项（success_rate 和 complete_rate）
            # 修正逻辑错误：应该是 and 而不是 or
            if name != "success_rate" or name != "complete_rate":
                # episode_sums: 每个奖励项在当前 episode 中的累计值
                # 用于在 episode 结束时计算平均奖励和记录日志
                self.episode_sums[name] += rew
                
        # ========== 步骤3：奖励裁剪（可选）==========        
        # 如果配置了只使用正奖励，则将负奖励裁剪为0
        # 这种设置可以避免智能体学到"不做任何动作"的消极策略
        if self.cfg.rewards.only_positive_rewards:
            # torch.clip(x, min=0.): 将所有负值设为0，正值保持不变
            # 例如：[-0.5, 0.3, -0.1, 0.8] → [0.0, 0.3, 0.0, 0.8]
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # ========== 步骤4：添加终止奖励（在裁剪后）==========
        # 终止奖励是特殊的奖励项，只在 episode 结束时给出
        # 在奖励裁剪后添加，确保终止奖励（通常是负值）不被裁剪掉
        if "termination" in self.reward_scales:
            # _reward_termination(): 计算终止奖励
            # 通常在机器人摔倒、超时、或违反约束时给出负奖励
            rew = self._reward_termination() * self.reward_scales["termination"]
            
            # 添加到总奖励（不受 only_positive_rewards 影响）
            self.rew_buf += rew
            
            # 累加到 episode 统计
            self.episode_sums["termination"] += rew
            
        # ========== 奖励计算示例 ==========
        """
        假设某个时间步的奖励计算：
        
        奖励项及其值：
        - lin_vel_z: 0.1 (垂直速度偏离惩罚)
        - orientation: 0.05 (姿态偏离惩罚) 
        - tracking_lin_vel: 0.8 (线速度跟踪奖励)
        - torques: 0.2 (力矩惩罚)
        
        权重设置：
        - lin_vel_z: -2.0 (惩罚权重)
        - orientation: -1.0 (惩罚权重)
        - tracking_lin_vel: 1.5 (奖励权重)
        - torques: -0.1 (小惩罚权重)
        
        计算过程：
        1. rew_buf = 0
        2. rew_buf += 0.1 * (-2.0) = -0.2
        3. rew_buf += 0.05 * (-1.0) = -0.25
        4. rew_buf += 0.8 * 1.5 = 0.95  
        5. rew_buf += 0.2 * (-0.1) = 0.93
        
        最终总奖励：0.93
        
        如果 only_positive_rewards=True 且总奖励为负，则会被裁剪为0
        """
    
    def compute_observations(self):
        """ 
        计算观测值（本体感知）
        组装机器人的感受器观测，包括IMU、关节状态、命令信息、接触状态等
        这些观测将作为强化学习策略网络的输入
        """
        
        # ========== 步骤1：计算IMU观测（惯性测量单元）==========
        # 提取机器人的姿态信息，只包含roll和pitch角度
        # 不包含yaw角是因为朝向信息通过其他方式提供（delta_yaw）
        # imu_obs: 形状为 [num_envs, 2] 的张量
        # self.roll: 滚转角，绕X轴旋转，范围 [-π, π]
        # self.pitch: 俯仰角，绕Y轴旋转，范围 [-π, π]
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        
        # ========== 步骤2：定期更新朝向误差信息 ==========
        # 每5个时间步更新一次朝向误差，减少计算开销
        if self.global_counter % 5 == 0:
            # 计算当前目标点的朝向误差
            # self.target_yaw: 指向当前目标点的朝向角
            # self.yaw: 机器人当前的朝向角
            self.delta_yaw = self.target_yaw - self.yaw
            
            # 计算下一个目标点的朝向误差
            # 提供更远的导航信息，帮助机器人规划路径
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        
        # ========== 步骤3：组装本体感受观测向量 ==========
        # 将各种传感器信息拼接成一个观测向量
        obs_buf = torch.cat((
            # 3维：基座角速度（本体坐标系）
            # 乘以缩放因子进行归一化，提高训练稳定性
            self.base_ang_vel * self.obs_scales.ang_vel,   # [num_envs, 3]
            
            # 2维：IMU姿态信息（roll, pitch）
            # 不包含yaw是因为朝向通过delta_yaw提供
            imu_obs,    # [num_envs, 2]
            
            # 1维：占位符（暂时不使用的朝向误差）
            # 乘以0表示该信息被屏蔽，可能用于调试或实验
            0 * self.delta_yaw[:, None],  # [num_envs, 1]
            
            # 1维：当前目标点朝向误差
            # 告诉机器人应该朝哪个方向转动才能面向目标点
            self.delta_yaw[:, None],  # [num_envs, 1]
            
            # 1维：下一个目标点朝向误差  
            # 提供更远的导航信息，帮助路径规划
            self.delta_next_yaw[:, None],  # [num_envs, 1]
            
            # 2维：占位符（暂时不使用的速度命令）
            # 原本可能包含前向和侧向速度命令
            0 * self.commands[:, 0:2],  # [num_envs, 2]
            
            # 1维：前向速度命令
            # 告诉机器人应该以多快的速度前进
            self.commands[:, 0:1],  # [num_envs, 1]
            
            # 2维：环境类型编码（one-hot）
            # 区分不同类型的地形或任务
            # env_class=17可能表示特殊的地形类型
            (self.env_class != 17).float()[:, None],  # [num_envs, 1] 非17类型
            (self.env_class == 17).float()[:, None],   # [num_envs, 1] 17类型
            
            # 19维：关节位置偏差（对于H1机器人）
            # (当前关节位置 - 默认关节位置) × 缩放因子
            # 告诉策略网络关节偏离中性位置的程度
            (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,  # [num_envs, 19]
            
            # 19维：关节速度（对于H1机器人）
            # 乘以缩放因子进行归一化
            self.dof_vel * self.obs_scales.dof_vel,  # [num_envs, 19]
            
            # 19维：上一步的动作命令（对于H1机器人）
            # 提供动作历史信息，有助于动作平滑性
            self.action_history_buf[:, -1],  # [num_envs, 19]
            
            # 2维：脚部接触状态
            # contact_filt: 布尔类型的接触状态，转换为float并减去0.5
            # 将[0,1]映射到[-0.5,0.5]，使数据以0为中心
            self.contact_filt.float() - 0.5,  # [num_envs, 2]
        ), dim=-1)  # 沿最后一个维度拼接
        
        # ========== 步骤4：构建特权观测（仿真中可获得，现实中不可获得）==========
        
        # 显式特权信息：机器人的线速度（本体坐标系）
        # 在现实中需要通过状态估计获得，仿真中可以直接读取
        # 重复3次可能是为了匹配某种网络结构的输入维度要求
        priv_explicit = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # [num_envs, 3] 实际线速度
            0 * self.base_lin_vel,                        # [num_envs, 3] 占位符
            0 * self.base_lin_vel,                        # [num_envs, 3] 占位符
        ), dim=-1)  # 总共 [num_envs, 9]
        
        # 潜在特权信息：物理参数（域随机化参数）
        # 这些参数在现实中很难准确获得，但在仿真中已知
        priv_latent = torch.cat((
            self.mass_params_tensor,      # [num_envs, 4] 质量和质心参数
            self.friction_coeffs_tensor,  # [num_envs, 1] 摩擦系数
            self.motor_strength[0] - 1,   # [num_envs, 19] 电机强度参数P（减1归一化）
            self.motor_strength[1] - 1    # [num_envs, 19] 电机强度参数D（减1归一化）
        ), dim=-1)  # 总共 [num_envs, 4+1+19+19=43]
        
        # ========== 步骤5：组装完整观测向量 ==========
        if self.cfg.terrain.measure_heights:
            # 计算相对地形高度
            # root_states[:, 2]: 机器人当前高度
            # 0.3: 参考高度偏移
            # measured_heights: 周围地形高度采样点
            # clip到[-1,1]: 限制高度差范围，提高训练稳定性
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            
            # 拼接所有观测：本体感受 + 地形高度 + 特权观测 + 历史观测
            self.obs_buf = torch.cat([
                obs_buf,                                           # 本体感受观测
                heights,                                           # 地形高度信息
                priv_explicit,                                     # 显式特权观测
                priv_latent,                                       # 潜在特权观测
                self.obs_history_buf.view(self.num_envs, -1)      # 历史观测（展平）
            ], dim=-1)
        else:
            # 不使用地形高度时的观测拼接
            self.obs_buf = torch.cat([
                obs_buf,                                           # 本体感受观测
                priv_explicit,                                     # 显式特权观测
                priv_latent,                                       # 潜在特权观测
                self.obs_history_buf.view(self.num_envs, -1)      # 历史观测（展平）
            ], dim=-1)
        
        # ========== 步骤6：屏蔽特定观测维度 ==========
        # 将第6-7维（索引6:8）设为0，可能是为了屏蔽某些不需要的信息
        # 这可能对应于之前被乘以0的命令维度
        obs_buf[:, 6:8] = 0  

        # ========== 步骤7：更新观测历史缓冲区 ==========
        # 维护一个滑动窗口的观测历史，用于提供时序信息
        self.obs_history_buf = torch.where(
            # 条件：episode长度<=1（刚开始或刚重置）
            (self.episode_length_buf <= 1)[:, None, None], 
            # 真值：用当前观测填充整个历史缓冲区
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            # 假值：滑动窗口更新（移除最旧的，添加最新的）
            torch.cat([
                self.obs_history_buf[:, 1:],  # 移除最旧的观测
                obs_buf.unsqueeze(1)          # 添加当前观测
            ], dim=1)
        )

        # ========== 步骤8：更新接触历史缓冲区 ==========
        # 维护脚部接触状态的历史信息
        self.contact_buf = torch.where(
            # 条件：episode长度<=1（刚开始或刚重置）
            (self.episode_length_buf <= 1)[:, None, None], 
            # 真值：用当前接触状态填充整个接触缓冲区
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            # 假值：滑动窗口更新
            torch.cat([
                self.contact_buf[:, 1:],                    # 移除最旧的接触状态
                self.contact_filt.float().unsqueeze(1)     # 添加当前接触状态
            ], dim=1)
        )
        
        # ========== 观测向量维度总结 ==========
        """
        最终观测向量的组成（以H1机器人为例）：
        
        本体感受观测 obs_buf：
        - 角速度: 3维
        - IMU姿态: 2维 (roll, pitch)
        - 朝向信息: 3维 (占位符1 + delta_yaw + delta_next_yaw)
        - 速度命令: 3维 (占位符2 + 前向速度命令1)
        - 环境类型: 2维 (one-hot编码)
        - 关节位置: 19维
        - 关节速度: 19维  
        - 历史动作: 19维
        - 接触状态: 2维
        小计: 72维
        
        特权观测：
        - 显式特权: 9维 (线速度相关)
        - 潜在特权: 43维 (物理参数)
        小计: 52维
        
        地形高度: 396维 (如果启用)
        历史观测: 72×history_len 维
        
        总维度: 72 + 52 + 396 + 72×history_len (具体取决于配置)
        """
            
    def get_noisy_measurement(self, x, scale):
        """
        为传感器测量值添加噪声
        
        在真实机器人中，传感器读数总是包含噪声。为了提高策略的鲁棒性，
        在仿真中模拟这种噪声是很重要的。
        
        Args:
            x (torch.Tensor): 原始测量值
            scale (float): 噪声缩放因子
            
        Returns:
            torch.Tensor: 添加噪声后的测量值
        """
        if self.cfg.noise.add_noise:
            # 生成[-1, 1]范围内的均匀随机噪声，然后乘以缩放因子和噪声级别
            noise = (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
            x = x + noise
        return x

    def create_sim(self):
        """ 
        创建仿真环境、地形和机器人环境
        
        这个方法是环境初始化的核心，包括以下步骤：
        1. 设置仿真的基本参数（坐标轴、图形设备等）
        2. 创建Isaac Gym仿真实例
        3. 根据配置创建地形（平面或复杂网格地形）
        4. 创建所有机器人环境实例
        """
        # 设置向上轴索引：2表示z轴向上，1表示y轴向上 -> 需要相应调整重力方向
        self.up_axis_idx = 2
        
        # 如果使用深度相机，即使在无头模式下也需要图形设备ID
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id
        
        # 创建Isaac Gym仿真实例
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # 开始计时，用于测量地形创建时间
        start = time()
        print("*"*80)
        
        # 获取地形网格类型配置
        mesh_type = terrain_config.mesh_type

        # 根据地形类型创建不同的地面
        if mesh_type=='None':
            # 创建简单的平面地面
            self._create_ground_plane()
        else:
            # 创建复杂的网格地形（如山丘、台阶、障碍物等）
            self.terrain = Terrain(self.num_envs)  # 初始化地形生成器
            self._create_trimesh()                 # 创建三角网格地形

        # 打印地形创建完成信息和耗时
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        
        # 创建所有机器人环境实例
        self._create_envs()

    def set_camera(self, position, lookat):
        """ 
        设置相机位置和朝向
        
        Args:
            position (list): 相机位置坐标 [x, y, z]
            lookat (list): 相机朝向的目标点坐标 [x, y, z]
        """
        # 将Python列表转换为Isaac Gym的Vec3格式
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])      # 相机位置
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])         # 相机目标点
        # 设置查看器相机的位置和朝向
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ 
        物理步骤后的回调函数
        
        在计算终止条件、奖励和观测之前调用的回调函数。
        默认行为包括：
        1. 根据目标和朝向计算角速度命令
        2. 计算测量的地形高度
        3. 随机推动机器人（域随机化）
        4. 重新采样运动命令
        """
        
        # 检查哪些环境需要重新采样命令（基于重采样时间间隔）
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())  # 为需要的环境重新采样命令

        # 🎯 朝向命令模式处理（heading_command = True）
        # if self.cfg.commands.heading_command:
        #     """
        #     朝向命令模式说明：
            
        #     ❓ 您的疑问解答：
        #     1. 是否启用：✅ 已启用（base_config中 heading_command = True）
        #     2. 角速度来源：并非直接来自目标点，而是来自随机采样的目标朝向
        #     3. 目标点的作用：用于奖励计算和观测，不直接影响运动命令
            
        #     🔄 工作流程：
        #     1. _resample_commands(): 随机采样目标朝向 → commands[:, 3]
        #     2. 此处：根据朝向误差计算角速度 → commands[:, 2]
        #     3. _update_goals(): 计算目标点朝向 → self.target_yaw（用于奖励，不覆盖commands）
            
        #     🎯 设计理念：
        #     - 目标朝向（commands[:, 3]）：高级导航指令，可以是任意方向
        #     - 目标点朝向（self.target_yaw）：具体路径导航，用于奖励和观测
        #     - 机器人需要学会在指定朝向下，同时导航到目标点
        #     """
            
        #     # ========== 步骤1：计算机器人当前的实际朝向 ==========
            
        #     # 1.1 将机器人本体坐标系的前向向量转换到世界坐标系
        #     # self.forward_vec = [1, 0, 0] 表示机器人本体的前向（X轴正方向）
        #     # self.base_quat 是机器人当前的姿态四元数，形状: [num_envs, 4]
        #     # 例如：如果机器人绕Z轴旋转了90度，四元数为 [0, 0, sin(π/4), cos(π/4)]
        #     forward = quat_apply(self.base_quat, self.forward_vec)
        #     # 结果示例：如果机器人朝向Y轴正方向，forward ≈ [0, 1, 0]
            
        #     # 1.2 从3D前向向量计算2D偏航角
        #     # 使用atan2计算向量在XY平面的角度，范围 [-π, π]
        #     # atan2(y, x) 返回从X轴到点(x,y)的角度
        #     heading = torch.atan2(forward[:, 1], forward[:, 0])
        #     # 例子：
        #     # - forward = [1, 0, 0] → heading = atan2(0, 1) = 0 rad (朝向X轴正方向)
        #     # - forward = [0, 1, 0] → heading = atan2(1, 0) = π/2 rad (朝向Y轴正方向)  
        #     # - forward = [-1, 0, 0] → heading = atan2(0, -1) = π rad (朝向X轴负方向)
        #     # - forward = [0, -1, 0] → heading = atan2(-1, 0) = -π/2 rad (朝向Y轴负方向)
            
        #     # ========== 步骤2：计算朝向误差和角速度命令 ==========
            
        #     # 2.1 计算朝向误差
        #     # self.commands[:, 3] 是目标朝向角度（在_resample_commands中随机采样）
        #     # 例如：目标朝向 = π/4 rad (45度)，当前朝向 = 0 rad (0度)
        #     heading_error = self.commands[:, 3] - heading
        #     # heading_error = π/4 - 0 = π/4 rad (需要向左转45度)
            
        #     # 2.2 将角度误差限制在 [-π, π] 范围内
        #     # 避免出现 ±2π 的大角度跳跃，选择最短的旋转路径
        #     heading_error_wrapped = wrap_to_pi(heading_error)
        #     # 例子：
        #     # - 如果误差 = 1.5π，wrap后 = -0.5π (向右转90度比向左转270度更短)
        #     # - 如果误差 = -1.5π，wrap后 = 0.5π (向左转90度比向右转270度更短)
            
        #     # 2.3 应用比例控制器计算角速度
        #     # 0.8 是控制增益Kp，决定响应速度和稳定性
        #     angular_velocity = 0.8 * heading_error_wrapped
        #     # 例如：heading_error = π/4，则 angular_velocity = 0.8 * π/4 ≈ 0.628 rad/s
            
        #     # 2.4 限制角速度命令的最大值
        #     # 防止过大的转向速度导致机器人失控或不稳定
        #     self.commands[:, 2] = torch.clip(angular_velocity, -1.0, 1.0)
        #     # 如果计算出的角速度 > 1.0 rad/s，则限制为 1.0 rad/s
        #     # 如果计算出的角速度 < -1.0 rad/s，则限制为 -1.0 rad/s
            
        #     # ========== 步骤3：死区处理（避免微小抖动）==========
            
        #     # 3.1 设置角速度命令的最小阈值
        #     # 当角速度命令太小时，设为0，避免不必要的微小转动
        #     # self.cfg.commands.ang_vel_clip 通常设为 0.1 rad/s
        #     small_command_mask = torch.abs(self.commands[:, 2]) <= self.cfg.commands.ang_vel_clip
        #     self.commands[:, 2] = torch.where(small_command_mask, 
        #                                     torch.zeros_like(self.commands[:, 2]), 
        #                                     self.commands[:, 2])
        #     # 等价于：self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
            
        #     # ========== 完整数值计算示例 ==========
        #     """
        #     假设有一个机器人：
            
        #     初始状态：
        #     - 当前朝向：0 rad (朝向X轴正方向)
        #     - 目标朝向：π/2 rad (朝向Y轴正方向，即向左转90度)
        #     - 控制增益：0.8
        #     - 角速度限制：[-1.0, 1.0] rad/s
        #     - 死区阈值：0.1 rad/s
            
        #     计算过程：
        #     1. heading_error = π/2 - 0 = π/2 ≈ 1.57 rad
        #     2. wrap_to_pi(1.57) = 1.57 rad (已在范围内)
        #     3. angular_velocity = 0.8 × 1.57 ≈ 1.256 rad/s
        #     4. clip到[-1,1]: min(max(1.256, -1), 1) = 1.0 rad/s
        #     5. 死区检查: |1.0| > 0.1，保持 1.0 rad/s
            
        #     结果：commands[:, 2] = 1.0 rad/s (最大向左转速度)
            
        #     随着机器人转动，heading逐渐接近π/2：
        #     - 当heading = π/4时，误差 = π/2 - π/4 = π/4，角速度 = 0.8×π/4 ≈ 0.628 rad/s
        #     - 当heading = π/2-0.1时，误差 ≈ 0.1，角速度 = 0.8×0.1 = 0.08 rad/s < 0.1，设为0
        #     - 机器人停止转动，达到目标朝向
        #     """
            
        #     # 计算机器人当前的前向方向向量（世界坐标系）
        #     # forward: 形状为 [num_envs, 3] 的张量，表示每个机器人在世界坐标系中的实际前进方向
        #     # 通过四元数旋转将机器人本体坐标系的前向向量 [1,0,0] 转换到世界坐标系
        #     # self.base_quat: 机器人基座的四元数姿态，形状为 [num_envs, 4]
        #     # self.forward_vec: 机器人本体前向向量 [1,0,0]，形状为 [num_envs, 3]
        #     forward = quat_apply(self.base_quat, self.forward_vec)
            
        #     # 计算当前朝向角度（偏航角）
        #     # heading: 形状为 [num_envs] 的张量，单位为弧度，范围 [-π, π]
        #     # 使用 atan2 函数计算前向向量在世界坐标系 XY 平面的角度
        #     # forward[:, 1]: Y 方向分量，forward[:, 0]: X 方向分量
        #     # 0 表示朝向正 X 轴方向，π/2 表示朝向正 Y 轴方向
        #     heading = torch.atan2(forward[:, 1], forward[:, 0])
            
        #     # 🧮 核心计算：朝向误差 → 角速度命令
        #     # target_heading - current_heading = heading_error
        #     # heading_error * gain = angular_velocity_command
            
        #     # 计算角速度命令，实现朝向控制
        #     # self.commands[:, 2]: 角速度命令，形状为 [num_envs]，单位 rad/s
        #     # self.commands[:, 3]: 目标朝向角度，在 _resample_commands() 中随机采样
        #     # wrap_to_pi(): 将角度差限制在 [-π, π] 范围内，避免 ±2π 的大角度跳跃
        #     # 0.8: 角速度控制增益（比例控制器增益），控制转向的敏感度和稳定性
        #     #      - 增益过大：转向过于敏感，可能导致震荡
        #     #      - 增益过小：转向反应迟钝，难以跟踪目标朝向
        #     # torch.clip(-1., 1.): 将角速度命令限制在 [-1, 1] rad/s 范围内，防止过大的转向速度
        #     self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            
        #     # 角速度命令的死区处理（避免微小抖动）
        #     # self.cfg.commands.ang_vel_clip: 角速度命令的最小阈值，通常为 0.1 rad/s
        #     # 当计算出的角速度命令绝对值小于阈值时，将其设为 0
        #     # 这样可以避免机器人在接近目标朝向时产生不必要的微小转动和抖动
        #     # 提高控制的稳定性和能耗效率
        #     self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
        
        # 如果启用地形高度测量，定期更新高度数据
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights, self.measured_heights_data = self._get_heights()  # 获取机器人周围的地形高度
        
        # 如果启用机器人推动（域随机化），定期随机推动机器人
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()  # 随机施加外力扰动
    
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """
        为指定环境重新采样运动命令
        使用智能速度生成策略
        """
        self._resample_commands_intelligent(env_ids)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """
        智能课程学习：结合地形复杂度和移动距离动态调整难度
        
        根据机器人的移动距离和地形复杂度自动调整地形难度：
        - 如果机器人移动距离超过期望的80%，增加难度
        - 如果机器人移动距离低于期望的40%，降低难度
        - 完成最高难度的机器人会被随机分配到不同难度
        
        Args:
            env_ids (List[int]): 需要重置的环境ID列表
        """
        if not self.init_done:
            return

        # 1. 计算机器人从起始位置移动的距离
        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        
        # 2. 获取平均地形复杂度（使用历史缓冲区）
        max_steps = 20  # 最多统计最近20步的复杂度
        ptrs = self.terrain_complexity_ptr[env_ids]
        history_len = self.terrain_complexity_history.shape[1]
        
        # 计算每个环境的实际可用长度（防止未填满）
        valid_lens = torch.clamp(ptrs, max=max_steps)
        
        # 使用向量化操作计算平均复杂度（避免Python循环）
        avg_complexity = torch.zeros(len(env_ids), device=self.device)
        
        for i, eid in enumerate(env_ids):
            end = ptrs[i].item()
            start = max(0, end - valid_lens[i].item())
            
            # 环形缓冲区处理
            if end > start:
                vals = self.terrain_complexity_history[eid, start:end]
            else:
                vals = torch.cat([
                    self.terrain_complexity_history[eid, start:history_len],
                    self.terrain_complexity_history[eid, 0:end]
                ])
            
            avg_complexity[i] = vals.mean() if vals.numel() > 0 else 0.0
        
        # 3. 综合评估分数：距离 × (1 + 复杂度)
        performance_score = dis_to_origin * (1 + avg_complexity)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s * 1.2
        
        # 4. 晋级/降级判定
        move_up = performance_score > threshold * 0.8    # 移动距离超过期望的80% → 增加难度
        move_down = performance_score < threshold * 0.4  # 移动距离低于期望的40% → 降低难度
        
        # 5. 更新地形难度级别
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        
        # 6. 保持难度在合理范围
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        
        # 7. 更新环境类别和目标
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        # 8. 更新目标点
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        
        # 9. 更新当前和下一个目标点坐标
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)


    def _init_buffers(self):
        """
        初始化PyTorch张量缓冲区
        创建包含仿真状态和处理量的张量，用于高效的GPU计算
        """
        # 从Isaac Gym获取GPU状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)      # 获取机器人根部状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)            # 获取关节状态张量
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)   # 获取接触力张量
        # force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)    # 力传感器张量（暂未使用）
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 获取刚体状态张量

        # 刷新张量数据，确保获取最新的仿真状态
        self.gym.refresh_dof_state_tensor(self.sim)           # 刷新关节状态
        self.gym.refresh_actor_root_state_tensor(self.sim)    # 刷新根部状态
        self.gym.refresh_net_contact_force_tensor(self.sim)   # 刷新接触力
        self.gym.refresh_rigid_body_state_tensor(self.sim)    # 刷新刚体状态
        # self.gym.refresh_force_sensor_tensor(self.sim)     # 刷新力传感器（暂未使用）
            
        # 创建包装张量，便于不同数据切片的访问
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 根部状态：位置、姿态、线速度、角速度
        # 刚体状态：每个刚体的位置、姿态、线速度、角速度 (13维)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # 关节状态：位置和速度 (2维)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)

        # 创建常用状态的快捷访问
        self.dof_pos = self.dof_state[...,0]      # 关节位置
        self.dof_vel = self.dof_state[..., 1]     # 关节速度
        self.base_quat = self.root_states[:, 3:7] # 基座四元数姿态

        # 接触力：每个刚体在xyz轴上的接触力
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # 初始化后续使用的数据
        self.common_step_counter = 0  # 通用步数计数器
        self.extras = {}              # 额外信息字典
        
        # 重力向量：根据上轴方向设置重力方向
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 前向向量：机器人前进方向
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        # 初始化控制相关张量
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 关节力矩
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)                # 比例增益
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)                # 微分增益
        
        # 初始化动作相关张量
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)      # 当前动作
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 上一步动作
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 上上步动作
        
        # 初始化历史状态张量
        self.last_dof_vel = torch.zeros_like(self.dof_vel)                    # 上一步关节速度
        self.last_torques = torch.zeros_like(self.torques)                   # 上一步力矩
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])     # 上一步根部速度
        self.last_foot_action = torch.zeros_like(self.rigid_body_states[:, self.feet_indices, :])  # 上一步脚部动作

        # 目标到达计时器
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # 电机强度随机化（域随机化的一部分）
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        
        # 历史编码缓冲区（如果启用）
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        
        # 动作历史缓冲区（用于动作延迟）
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        
        # 接触历史缓冲区
        # self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        # 运动命令：线速度x、y和角速度yaw
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        # 重新采样所有环境的命令
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        # 命令缩放因子
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        
        # 脚部腾空时间和接触状态
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)  # 脚部腾空时间
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)        # 上一步接触状态
        
        # 基座在本体坐标系下的速度
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])   # 线速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 角速度
        # 投影重力（本体坐标系下的重力方向）
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 如果启用高度测量，初始化高度采样点
        if self.cfg.terrain.measure_heights:
            self.height_points, self.height_points_data = self._init_height_points()
        self.measured_heights = 0  # 测量的高度值
        self.measured_heights = 0  # 重复行（可能是代码错误）

        # 关节位置偏移和PD增益设置
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)          # 默认关节位置
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 所有环境的默认关节位置
        
        # 遍历所有关节，设置默认位置和PD增益
        for i in range(self.num_dofs):
            name = self.dof_names[i]  # 关节名称
            angle = self.cfg.init_state.default_joint_angles[name]  # 从配置获取默认角度
            self.default_dof_pos[i] = angle
            found = False
            # 为每个关节设置PD增益
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:  # 匹配关节名称
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # 设置比例增益
                    self.d_gains[i] = self.cfg.control.damping[dof_name]    # 设置微分增益
                    found = True
            if not found:  # 如果没有找到对应的增益设置
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        
        # 为所有环境复制默认关节位置
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # 增加批次维度
        self.default_dof_pos_all[:] = self.default_dof_pos[0]     # 复制到所有环境

        # 高度更新间隔设置
        self.height_update_interval = 1  # 默认每步更新
        if hasattr(self.cfg.env, "height_update_dt"):
            # 根据配置的时间间隔计算更新间隔步数
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        # 如果使用深度相机，初始化深度缓冲区
        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len,      # 缓冲区长度
                                            self.cfg.depth.resized[1],      # 图像高度
                                            self.cfg.depth.resized[0]).to(self.device)  # 图像宽度

        # 添加地形复杂度历史缓冲区初始化
        self.terrain_complexity_history = torch.zeros(self.num_envs, 100, device=self.device, dtype=torch.float)
        self.terrain_complexity_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _prepare_reward_function(self):
        """
        准备奖励函数列表
        查找所有非零奖励权重对应的奖励函数，用于计算总奖励
        函数名格式：self._reward_<REWARD_NAME>，其中<REWARD_NAME>是配置中奖励权重的名称
        """
        # 移除零权重的奖励项，并将非零权重乘以时间步长
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)  # 移除零权重项
            else:
                self.reward_scales[key] *= self.dt  # 乘以时间步长进行归一化
                
        # 准备奖励函数列表
        self.reward_functions = []  # 奖励函数列表
        self.reward_names = []      # 奖励名称列表
        
        for name, scale in self.reward_scales.items():
            if name=="termination":  # 跳过终止奖励（特殊处理）
                continue
            self.reward_names.append(name)
            name = '_reward_' + name  # 构建函数名
            # 通过反射获取奖励函数并添加到列表
            self.reward_functions.append(getattr(self, name))

        # 初始化episode奖励累计器
        # 为每个奖励项创建累计张量，用于跟踪每个环境的奖励总和
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ 
        向仿真中添加平面地面，根据配置设置摩擦力和恢复系数
        
        创建一个简单的水平平面作为机器人的行走表面。
        这是最基础的地形类型，适用于基本的步行训练。
        """
        # 创建平面参数对象
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)                    # 平面法向量（z轴向上）
        plane_params.static_friction = self.cfg.terrain.static_friction      # 静摩擦系数
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction    # 动摩擦系数
        plane_params.restitution = self.cfg.terrain.restitution             # 恢复系数（弹性）
        
        # 将地面平面添加到仿真中
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ 
        向仿真中添加三角网格地形，根据配置设置参数
        
        创建复杂的3D地形，包括山丘、台阶、障碍物等。
        注意：当horizontal_scale很小时，此方法会非常慢。
        
        这种地形可以提供更丰富的训练场景，提高机器人的适应性。
        """
        # 创建三角网格参数对象
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]      # 顶点数量
        tm_params.nb_triangles = self.terrain.triangles.shape[0]    # 三角形数量

        # 设置地形的位置变换（考虑边界大小）
        tm_params.transform.p.x = -self.terrain.cfg.border_size     # X轴偏移
        tm_params.transform.p.y = -self.terrain.cfg.border_size     # Y轴偏移
        tm_params.transform.p.z = 0.0                               # Z轴偏移（地面高度）
        
        # 设置物理属性
        tm_params.static_friction = self.cfg.terrain.static_friction    # 静摩擦系数
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 动摩擦系数
        tm_params.restitution = self.cfg.terrain.restitution           # 恢复系数（弹性）
        
        print("Adding trimesh to simulation...")
        # 将三角网格添加到仿真中（顶点和三角形数据需要按C顺序展平）
        self.gym.add_triangle_mesh(self.sim, 
                                 self.terrain.vertices.flatten(order='C'), 
                                 self.terrain.triangles.flatten(order='C'), 
                                 tm_params)  
        print("Trimesh added")
        
        # 将高度采样数据转换为PyTorch张量并移动到指定设备
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        # 将边缘掩码转换为PyTorch张量（用于检测机器人是否接近地形边缘）
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        """
        为指定的机器人环境附加深度相机
        
        这个方法在每个机器人上安装一个深度相机，用于视觉感知和观测。
        相机会跟随机器人移动，提供第一人称视角的深度信息。
        
        Args:
            i (int): 环境索引
            env_handle: 环境句柄
            actor_handle: 机器人actor句柄
        """
        if self.cfg.depth.use_camera:
            # 获取深度相机配置
            config = self.cfg.depth
            
            # 创建相机属性对象
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]           # 图像宽度（像素）
            camera_props.height = self.cfg.depth.original[1]          # 图像高度（像素）
            camera_props.enable_tensors = True                        # 启用张量输出（用于GPU加速）
            camera_horizontal_fov = self.cfg.depth.horizontal_fov     # 水平视野角度
            camera_props.horizontal_fov = camera_horizontal_fov

            # 在环境中创建相机传感器
            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)  # 保存相机句柄
            
            # 创建相机的本地变换（相对于机器人的位置和姿态）
            local_transform = gymapi.Transform()
            
            # 设置相机位置（相对于机器人根部）
            camera_position = np.copy(config.position)  # [x, y, z] 相对位置
            # 随机化相机俯仰角（增加训练多样性）
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            # 设置相机的位置和旋转
            local_transform.p = gymapi.Vec3(*camera_position)  # 位置向量
            # 设置相机旋转：绕y轴旋转（俯仰角），0表示roll和yaw
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            
            # 获取机器人根部刚体句柄
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            # print("rigid_body_names=",self.gym.get_actor_rigid_body_names(env_handle, actor_handle))

            # 将相机附加到机器人根部，使其跟随机器人移动
            # FOLLOW_TRANSFORM 表示相机会跟随刚体的变换
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        # print("rigid_body_names=",self.gym.get_actor_rigid_body_names(env_handle, actor_handle))

    def _create_envs(self):
        """ 
        创建所有机器人环境实例
        
        这是环境创建的核心方法，执行以下步骤：
        1. 加载机器人URDF/MJCF资产文件
        2. 为每个环境：
           2.1 创建环境实例
           2.2 调用DOF和刚体形状属性回调函数
           2.3 使用这些属性创建actor并添加到环境中
        3. 存储机器人不同部位的索引信息
        """
        # 构建机器人资产文件的完整路径
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)    # 资产文件所在目录
        asset_file = os.path.basename(asset_path)   # 资产文件名

        # 配置资产加载选项
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode          # 默认DOF驱动模式
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints            # 是否合并固定关节
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule  # 用胶囊体替换圆柱体
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments        # 是否翻转视觉附件
        asset_options.fix_base_link = self.cfg.asset.fix_base_link                           # 是否固定基座链接
        asset_options.density = self.cfg.asset.density                                       # 密度
        asset_options.angular_damping = self.cfg.asset.angular_damping                       # 角阻尼
        asset_options.linear_damping = self.cfg.asset.linear_damping                         # 线性阻尼
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity             # 最大角速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity               # 最大线速度
        asset_options.armature = self.cfg.asset.armature                                     # 电枢
        asset_options.thickness = self.cfg.asset.thickness                                   # 厚度
        asset_options.disable_gravity = self.cfg.asset.disable_gravity                       # 是否禁用重力

        # 加载机器人资产并获取基本信息
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)              # 获取DOF数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)    # 获取刚体数量
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)      # 获取DOF属性
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)  # 获取刚体形状属性

        # 从资产中保存刚体和DOF名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)  # 获取刚体名称列表
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)     # 获取DOF名称列表
        # print("DOF names:", self.dof_names)
        self.num_bodies = len(body_names)    # 刚体数量
        self.num_dofs = len(self.dof_names)  # DOF数量
        
        # 根据配置查找特定部位的名称
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]    # 脚部刚体名称
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]    # 膝盖刚体名称
        
        # 查找需要惩罚接触的刚体名称
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        
        # 查找接触后需要终止的刚体名称
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 设置机器人基座的初始状态
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()                        # 创建起始姿态变换
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])  # 设置起始位置

        # 获取环境原点位置
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)  # 环境边界下限
        env_upper = gymapi.Vec3(0., 0., 0.)  # 环境边界上限
        
        # 初始化存储列表和张量
        self.actor_handles = []    # 存储所有actor句柄
        self.envs = []            # 存储所有环境句柄
        self.cam_handles = []     # 存储所有相机句柄
        self.cam_tensors = []     # 存储所有相机张量
        # 质量参数张量：每个环境4个参数（质量、质心x、质心y、质心z）
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        # 为每个环境创建机器人实例
        for i in tqdm(range(self.num_envs)):
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            
            # 设置机器人起始位置
            pos = self.env_origins[i].clone()
            
            # 可选：随机化起始位置
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            
            # 可选：随机化起始偏航角
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            
            # 更新起始姿态位置
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            # 处理刚体形状属性（应用域随机化等）
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            
            # 创建机器人actor
            humanoid_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "Humanoid", i, self.cfg.asset.self_collisions, 0)
            
            # 处理DOF属性（关节刚度、阻尼等）
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, humanoid_handle, dof_props)
            
            # 处理刚体属性（质量、质心等）
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, humanoid_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, humanoid_handle, body_props, recomputeInertia=True)
            
            # 保存环境和actor句柄
            self.envs.append(env_handle)
            self.actor_handles.append(humanoid_handle)
            
            # 如果使用深度相机，为此环境附加相机
            self.attach_camera(i, env_handle, humanoid_handle)

            # 保存质量参数到张量
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        # 如果启用了摩擦力随机化，将摩擦系数转换为张量
        # print("open=",self.cfg.domain_rand.randomize_friction)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        # print("name=",feet_names)

        # 获取脚部刚体的索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # 获取膝盖刚体的索引
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        # 获取需要惩罚接触的刚体索引
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        # 获取接触后需要终止的刚体索引
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
 
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if terrain_config.mesh_type == "None":
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
        else:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level # 2
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

    def _parse_cfg(self, cfg):
        """
        解析配置文件，提取并计算关键参数
        将配置文件中的参数转换为环境运行时需要的格式
        
        Args:
            cfg: 配置对象，包含所有环境和训练参数
        """
        # 计算控制时间步长 = 控制频率降采样 × 仿真时间步长
        # 例如：decimation=4, sim_dt=0.005s → control_dt=0.02s (50Hz控制频率)
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        
        # 获取观测值归一化缩放因子
        # 用于将不同量纲的观测值归一化到相似范围
        self.obs_scales = self.cfg.normalization.obs_scales
        
        # 将奖励权重配置转换为字典格式，便于后续处理
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        
        # 奖励归一化因子（当前设为1，不进行归一化）
        # 可以设为所有奖励权重之和来归一化总奖励
        reward_norm_factor = 1  # np.sum(list(self.reward_scales.values()))
        
        # 对每个奖励项进行归一化处理
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
            
        # 根据是否启用课程学习选择命令范围
        if self.cfg.commands.curriculum:
            # 课程学习模式：使用渐进式命令范围
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            # 固定模式：直接使用最大命令范围
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)

        # 设置episode长度参数
        self.max_episode_length_s = self.cfg.env.episode_length_s  # episode最大时长（秒）
        # 将时间长度转换为控制步数
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # 计算域随机化中推力施加的间隔步数
        # 将时间间隔转换为控制步数间隔
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(255, 0, 0))
        i = self.lookat_id  
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        if self.save:
            heights = self.measured_heights_data[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points_data[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(2):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)

        # only for recording dataset, not for policy
        y_data = torch.tensor(self.cfg.terrain.dataset_points_y, device=self.device, requires_grad=False)
        x_data = torch.tensor(self.cfg.terrain.dataset_points_x, device=self.device, requires_grad=False)
        grid_x_data, grid_y_data = torch.meshgrid(x_data, y_data)
        self.num_height_points_data = grid_x_data.numel()
        points_data = torch.zeros(self.num_envs, self.num_height_points_data, 3, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]

            # visualize saved height point
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points_data,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points_data,2), device=self.device).squeeze() + offset
            points_data[i, :, 0] = grid_x_data.flatten() #+ xy_noise[:, 0]
            points_data[i, :, 1] = grid_y_data.flatten() #+ xy_noise[:, 1]
        return points, points_data

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            points_data = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points_data), self.height_points_data[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
            points_data = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points_data), self.height_points_data) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        points_data += self.terrain.cfg.border_size
        points_data = (points_data/self.terrain.cfg.horizontal_scale).long()
        px_data = points_data[:, :, 0].view(-1)
        py_data = points_data[:, :, 1].view(-1)
        px_data = torch.clip(px_data, 0, self.height_samples.shape[0]-2)
        py_data = torch.clip(py_data, 0, self.height_samples.shape[1]-2)
        heights1_data = self.height_samples[px_data, py_data]
        heights2_data = self.height_samples[px_data+1, py_data]
        heights3_data = self.height_samples[px_data, py_data+1]
        heights_data = torch.min(heights1_data, heights2_data)
        heights_data = torch.min(heights_data, heights3_data)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale, heights_data.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    def _resample_commands_intelligent(self, env_ids):
        """智能的命令重采样（策略1.2.4）"""
        
        #  基于高度信息生成自适应速度
        if self.cfg.commands.height_adaptive_speed:  # 新增配置开关
            adaptive_speeds = self._generate_adaptive_speed(env_ids)
            self.commands[env_ids, 0] = adaptive_speeds
        else:
            # 保留原有的随机采样作为备选
            self.commands[env_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0], 
                self.command_ranges["lin_vel_x"][1], 
                (len(env_ids), 1), device=self.device
            ).squeeze(1) 
        
        if self.cfg.commands.heading_command:
            # 1. 随机采样朝向角度
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], 
                self.command_ranges["heading"][1], 
                (len(env_ids), 1), 
                device=self.device
            ).squeeze(1)
            
            # 2. 计算当前朝向角度
            forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            
            # 3. 计算朝向误差和角速度命令
            heading_error = self.commands[env_ids, 3] - heading
            heading_error_wrapped = wrap_to_pi(heading_error)
            
            # 4. 应用比例控制器计算角速度
            angular_velocity = 0.8 * heading_error_wrapped
            self.commands[env_ids, 2] = torch.clip(angular_velocity, -1.0, 1.0)
            
            # 5. 死区处理（避免微小抖动）
            small_command_mask = torch.abs(self.commands[env_ids, 2]) <= self.cfg.commands.ang_vel_clip
            self.commands[env_ids, 2] = torch.where(small_command_mask, 
                                                   torch.zeros_like(self.commands[env_ids, 2]), 
                                                   self.commands[env_ids, 2])
        else:
            # 传统模式：直接采样角速度命令
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], 
                self.command_ranges["ang_vel_yaw"][1], 
                (len(env_ids), 1), 
                device=self.device
            ).squeeze(1)
            
            # 如果角速度太小，设为0（避免微小抖动）
            small_ang_vel_mask = torch.abs(self.commands[env_ids, 2]) <= self.cfg.commands.ang_vel_clip
            self.commands[env_ids, 2] = torch.where(small_ang_vel_mask, 
                                                   torch.zeros_like(self.commands[env_ids, 2]), 
                                                   self.commands[env_ids, 2])

        # 如果前向速度太小，将前向和侧向速度都设为0
        small_lin_vel_mask = torch.abs(self.commands[env_ids, 0]) <= self.cfg.commands.lin_vel_clip
        self.commands[env_ids, 0] = torch.where(small_lin_vel_mask, 
                                               torch.zeros_like(self.commands[env_ids, 0]), 
                                               self.commands[env_ids, 0])
        self.commands[env_ids, 1] = torch.where(small_lin_vel_mask, 
                                               torch.zeros_like(self.commands[env_ids, 1]), 
                                               self.commands[env_ids, 1])


    def _generate_adaptive_speed(self, env_ids):
        """基于地形复杂度生成自适应速度"""
        complexity = self._analyze_terrain_complexity()[env_ids]
        
        # 速度策略：
        # - 简单地形（complexity < 0.3）：高速前进 [1.0, 1.5] m/s
        # - 中等地形（0.3 ≤ complexity < 0.7）：中速前进 [0.5, 1.0] m/s  
        # - 困难地形（complexity ≥ 0.7）：低速前进 [0.2, 0.5] m/s
        
        base_speed = 1.0 - complexity  # 基础速度：1.5 → 0.5
        speed_range = 0.3 * (1 - complexity)  # 速度范围：简单地形变化大，困难地形变化小
        
        # 在基础速度±范围内随机采样
        min_speed = torch.clamp(base_speed - speed_range, 0.1, 1.4)
        max_speed = torch.clamp(base_speed + speed_range, 0.2, 1.5)
        
        adaptive_speeds = torch.empty((len(env_ids), 1), device=self.device)
        adaptive_speeds.uniform_(0, 1)
        adaptive_speeds = min_speed.unsqueeze(1) + adaptive_speeds * (max_speed.unsqueeze(1) - min_speed.unsqueeze(1))
        adaptive_speeds = adaptive_speeds.squeeze(1)
        
        return adaptive_speeds


    def _analyze_terrain_complexity(self):
        """分析前方地形复杂度"""
        # 检查measured_heights是否已初始化
        if not hasattr(self, 'measured_heights') or isinstance(self.measured_heights, int):
            # 如果还没有初始化，返回默认复杂度
            return torch.zeros(self.num_envs, device=self.device)
        
        front_x_indices = [3, 4, 5, 6]  # x = 0, 0.15, 0.3, 0.45 的索引
        front_point_indices = []
        for x_idx in front_x_indices:
            for y_idx in range(11):  # 所有y方向
                front_point_indices.append(x_idx * 11 + y_idx)
        
        forward_heights = self.measured_heights[:, front_point_indices]
        
        # 计算每一行的方差、最大最小值和粗糙度
        height_variance = torch.zeros(self.num_envs, device=self.device)
        height_gradient = torch.zeros(self.num_envs, device=self.device)
        height_roughness = torch.zeros(self.num_envs, device=self.device)
        
        for i in range(4):  # 4行：x = 0, 0.15, 0.3, 0.45
            row_heights = forward_heights[:, i*11:(i+1)*11]  # 第i行的11个点
            
            # 每行的方差
            row_variance = torch.var(row_heights, dim=1)
            height_variance += row_variance
            
            # 每行的最大最小值差
            row_range = torch.max(row_heights, dim=1)[0] - torch.min(row_heights, dim=1)[0]
            height_gradient += row_range
            
            # 每行的粗糙度（相邻点的高度差）
            row_roughness = torch.mean(torch.abs(torch.diff(row_heights, dim=1)), dim=1)
            height_roughness += row_roughness
        
        # 取平均值
        height_variance /= 4
        height_gradient /= 4
        height_roughness /= 4
        
        complexity = torch.clamp(
            self.cfg.commands.speed_complexity_weight * height_variance + 
            self.cfg.commands.speed_gradient_weight * height_gradient + 
            self.cfg.commands.speed_roughness_weight * height_roughness,
            0.0, 1.0
        )
        return complexity

    def _update_terrain_complexity_history(self):
        """更新地形复杂度历史缓冲区"""
        # 计算当前地形复杂度
        current_complexity = self._analyze_terrain_complexity()
        
        # 将当前复杂度存储到历史缓冲区
        for env_id in range(self.num_envs):
            ptr = self.terrain_complexity_ptr[env_id]
            self.terrain_complexity_history[env_id, ptr] = current_complexity[env_id]
            
            # 更新指针（环形缓冲区）
            self.terrain_complexity_ptr[env_id] = (ptr + 1) % self.terrain_complexity_history.shape[1]
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_reach_goal(self):
        """到达目标奖励（指数衰减，与跟踪奖励一致）"""
        distance_to_goal = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)
        
        # 使用指数衰减，距离越近奖励越高
        return torch.exp(-distance_to_goal / 0.2)  # 0.5是衰减参数


    