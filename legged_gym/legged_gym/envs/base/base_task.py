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

import sys
from isaacgym import gymapi
from isaacgym import gymutil, gymtorch
import numpy as np
import torch
import time

# 强化学习任务的基类
class BaseTask():
    """
    强化学习任务基类
    
    这个类提供了所有RL环境的基础功能，包括：
    - Isaac Gym仿真环境的初始化
    - 观测和动作缓冲区的管理
    - 渲染和可视化功能
    - 键盘交互控制
    - 相机管理
    
    所有具体的机器人环境（如HumanoidRobot）都继承自这个基类
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """
        初始化基础任务环境
        
        Args:
            cfg: 环境配置对象（如H1_2FixCfg实例）
            sim_params: 仿真参数
            physics_engine: 物理引擎类型
            sim_device: 仿真设备（如'cuda:0'或'cpu'）
            headless: 是否无头模式（不显示图形界面）
        """
        # 获取Isaac Gym接口
        self.gym = gymapi.acquire_gym()

        # 保存仿真相关参数
        self.sim_params = sim_params        # 仿真参数
        self.physics_engine = physics_engine  # 物理引擎
        self.sim_device = sim_device        # 仿真设备
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)  # 解析设备类型和ID
        self.headless = headless            # 是否无头模式

        # 确定环境设备：只有当仿真在GPU上且使用GPU管道时，环境设备才是GPU
        # 否则PhysX会将张量复制到CPU
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device   # 使用GPU设备
        else:
            self.device = 'cpu'             # 使用CPU设备

        # 图形设备设置，-1表示不渲染
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1    # 无头模式不渲染

        # 从配置中获取环境参数
        self.num_envs = cfg.env.num_envs                    # 并行环境数量
        self.num_obs = cfg.env.num_observations             # 观测维度
        self.num_privileged_obs = cfg.env.num_privileged_obs  # 特权观测维度
        self.num_actions = cfg.env.num_actions              # 动作维度
        
        # PyTorch JIT优化标志（提高性能）
        torch._C._jit_set_profiling_mode(False)    # 关闭JIT性能分析
        torch._C._jit_set_profiling_executor(False)  # 关闭JIT执行器分析

        # 分配缓冲区 - 所有张量都在指定设备上创建
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)  # 观测缓冲区
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)  # 奖励缓冲区
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)  # 重置标志缓冲区（初始全为1，表示需要重置）
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # episode长度缓冲区
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # 超时标志缓冲区
        
        # 特权观测缓冲区（用于非对称训练，如果有的话）
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        # 额外信息字典
        self.extras = {}

        # 创建环境、仿真和查看器
        self.create_sim()                   # 创建仿真环境（在子类中实现）
        self.gym.prepare_sim(self.sim)      # 准备仿真

        # TODO: 从配置文件读取
        self.enable_viewer_sync = True      # 是否启用查看器同步
        self.viewer = None                  # 查看器对象

        # 如果不是无头模式，设置键盘快捷键和相机
        if self.headless == False:
            # 订阅键盘快捷键
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())  # 创建查看器
            
            # 注册各种键盘事件
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")  # ESC键退出
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")  # V键切换查看器同步
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_F, "free_cam")  # F键切换自由相机模式
            
            # 数字键0-8用于查看不同的机器人
            for i in range(9):
                self.gym.subscribe_viewer_keyboard_event(
                self.viewer, getattr(gymapi, "KEY_"+str(i)), "lookat"+str(i))
            
            # 方括号键用于切换观察的机器人
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")   # [键：上一个机器人
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")  # ]键：下一个机器人
            
            # 空格键暂停
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SPACE, "pause")
            
            # WASD键用于控制机器人运动命令
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "vx_plus")      # W键：增加前向速度
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "vx_minus")     # S键：减少前向速度
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "left_turn")    # A键：左转
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "right_turn")   # D键：右转
        
        # 相机控制相关变量
        self.free_cam = False               # 是否自由相机模式
        self.lookat_id = 0                 # 当前观察的机器人ID
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)  # 相机相对位置向量

    def get_observations(self):
        """
        获取观测数据
        
        Returns:
            torch.Tensor: 观测缓冲区，形状为(num_envs, num_obs)
        """
        return self.obs_buf
    
    def get_privileged_observations(self):
        """
        获取特权观测数据（用于非对称训练）
        
        Returns:
            torch.Tensor或None: 特权观测缓冲区，形状为(num_envs, num_privileged_obs)
        """
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """
        重置指定的机器人环境
        
        Args:
            env_ids (torch.Tensor): 需要重置的环境ID列表
        
        Note:
            这是一个抽象方法，需要在子类中实现具体的重置逻辑
        """
        raise NotImplementedError

    def reset(self):
        """
        重置所有机器人环境
        
        Returns:
            tuple: (观测, 特权观测)
        """
        # 重置所有环境
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # 执行一步零动作以获取初始观测
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """
        执行一步仿真
        
        Args:
            actions (torch.Tensor): 动作张量，形状为(num_envs, num_actions)
        
        Returns:
            tuple: 通常返回(观测, 奖励, 完成标志, 信息)
        
        Note:
            这是一个抽象方法，需要在子类中实现具体的步进逻辑
        """
        raise NotImplementedError

    def lookat(self, i):
        """
        将相机对准指定的机器人
        
        Args:
            i (int): 机器人环境ID
        """
        look_at_pos = self.root_states[i, :3].clone()  # 获取机器人根部位置
        cam_pos = look_at_pos + self.lookat_vec        # 计算相机位置
        self.set_camera(cam_pos, look_at_pos)          # 设置相机位置和朝向

    def render(self, sync_frame_time=True):
        """
        渲染一帧并处理用户交互
        
        Args:
            sync_frame_time (bool): 是否同步帧时间
        """
        if self.viewer:
            # 检查窗口是否关闭
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            
            # 如果不是自由相机模式，跟随指定机器人
            if not self.free_cam:
                self.lookat(self.lookat_id)
            
            # 处理键盘事件
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()  # 退出程序
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync  # 切换同步模式
                
                # 非自由相机模式下的控制
                if not self.free_cam:
                    # 数字键选择观察的机器人
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    
                    # 方括号键切换机器人
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs  # 上一个机器人
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs  # 下一个机器人
                        self.lookat(self.lookat_id)
                    
                    # WASD键控制机器人运动命令
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] += 0.2      # 增加前向速度
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] -= 0.2      # 减少前向速度
                    if evt.action == "left_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] += 0.5      # 左转
                    if evt.action == "right_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] -= 0.5      # 右转
                
                # F键切换自由相机模式
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        # 切换到自由相机时，设置到配置的默认位置
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
                
                # 空格键暂停
                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    # 暂停循环
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        # 在暂停状态下仍处理事件
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False  # 再次按空格键取消暂停
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()
                        
            # 获取仿真结果
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)  # 从GPU获取结果

            self.gym.poll_viewer_events(self.viewer)  # 轮询查看器事件
            
            # 图形步进
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)      # 更新图形
                self.gym.draw_viewer(self.viewer, self.sim, True)  # 绘制查看器
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)  # 同步帧时间
            else:
                self.gym.poll_viewer_events(self.viewer)
            
            # 更新相机跟随向量（非自由相机模式）
            if not self.free_cam:
                # 获取当前相机变换
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                # 更新相机相对位置向量
                self.lookat_vec = cam_trans - look_at_pos
            
