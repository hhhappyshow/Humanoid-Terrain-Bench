# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Multi-Teacher Distillation Algorithm - Based on rsl_rl_old framework
# 多教师蒸馏算法 - 基于rsl_rl_old框架

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from rsl_rl.modules import MultiTeacherStudent
from rsl_rl.storage import RolloutStorage


class MultiTeacherDistillation:
    """多教师蒸馏算法 - 基于rsl_rl_old的PPO框架
    
    该算法实现多教师知识蒸馏训练：
    1. 学生策略学习模仿多个教师的行为
    2. 根据地形类型自适应权重分配
    3. 结合行为克隆和多样性损失
    """
    
    def __init__(
        self,
        actor_critic: MultiTeacherStudent,
        num_learning_epochs: int = 2,
        num_mini_batches: int = 4,
        learning_rate: float = 5e-4,
        max_grad_norm: float = 1.0,
        # 蒸馏损失配置
        distillation_loss_coef: float = 1.0,
        behavior_cloning_coef: float = 0.5,
        diversity_loss_coef: float = 0.1,
        # 地形自适应权重
        terrain_adaptive_weights: bool = True,
        weight_temperature: float = 1.0,
        # 动作处理配置
        clip_actions: float = 1.2,
        action_scale: float = 0.25,
        # 其他配置
        device: str = 'cpu',
        **kwargs
    ):
        """初始化多教师蒸馏算法
        
        Args:
            actor_critic: 多教师学生网络
            num_learning_epochs: 学习轮数
            num_mini_batches: 小批次数量
            learning_rate: 学习率
            max_grad_norm: 梯度裁剪
            distillation_loss_coef: 蒸馏损失系数
            behavior_cloning_coef: 行为克隆损失系数
            diversity_loss_coef: 多样性损失系数
            terrain_adaptive_weights: 是否使用地形自适应权重
            weight_temperature: 权重温度参数
            clip_actions: 动作裁剪值
            action_scale: 动作缩放因子
            device: 设备
        """
        
        if kwargs:
            print("MultiTeacherDistillation.__init__ got unexpected arguments: " + str([key for key in kwargs.keys()]))
        
        self.device = device
        
        # 网络和优化器
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 训练参数
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        
        # 蒸馏损失配置
        self.distillation_loss_coef = distillation_loss_coef
        self.behavior_cloning_coef = behavior_cloning_coef
        self.diversity_loss_coef = diversity_loss_coef
        
        # 地形自适应权重
        self.terrain_adaptive_weights = terrain_adaptive_weights
        self.weight_temperature = weight_temperature
        
        # 动作处理配置
        self.clip_actions = clip_actions
        self.action_scale = action_scale
        
        # 存储和转换
        self.storage = None  # 稍后初始化
        self.transition = RolloutStorage.Transition()
        
        # 训练统计
        self.counter = 0
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """初始化经验存储"""
        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            actor_obs_shape, 
            critic_obs_shape, 
            action_shape, 
            self.device
        )
    
    def test_mode(self):
        """切换到测试模式"""
        self.actor_critic.eval()
    
    def train_mode(self):
        """切换到训练模式"""
        self.actor_critic.train()
    
    def process_actions(self, raw_actions):
        """处理动作，与humanoid_robot中的动作处理逻辑相同
        
        Args:
            raw_actions (torch.Tensor): 原始动作
            
        Returns:
            torch.Tensor: 处理后的动作（与humanoid_robot.step()方法中self.actions相同）
        """
        clip_actions = self.clip_actions / self.action_scale
        processed_actions = torch.clip(raw_actions, -clip_actions, clip_actions).to(self.device)
        return processed_actions
    
    def act(self, obs, critic_obs, info, hist_encoding=False):
        """执行动作（训练时收集经验）- 只训练学生策略"""
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # 学生策略执行动作（保持梯度用于训练）
        self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # 存储观测（不需要梯度）
        self.transition.observations = obs.detach()
        self.transition.critic_observations = critic_obs.detach()
        
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        """处理环境步骤"""
        rewards_total = rewards.clone()
        
        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        
        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        
        return rewards_total
    
    def compute_returns(self, last_critic_obs):
        """计算回报（简化版本，主要关注蒸馏损失）"""
        # 只需要学生网络的价值估计，不需要梯度
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_critic_obs.detach())
        
        # 注意：对于纯蒸馏，可能不需要传统的返回值计算
        # 但为了兼容现有框架，我们保留这个接口
        # self.storage.compute_returns(last_values, gamma=0.99, lam=0.95)
        
        # 简化版本：直接使用奖励作为目标
        for step in range(self.storage.num_transitions_per_env):
            self.storage.returns[step] = self.storage.rewards[step]
    
    def update(self):
        """更新网络参数 - 多教师蒸馏的核心"""
        mean_distillation_loss = 0
        mean_behavior_cloning_loss = 0
        mean_diversity_loss = 0
        mean_total_loss = 0
        
        # 调试信息：检查教师模型状态
        print(f"[DEBUG] 开始蒸馏更新，教师模型加载状态: {self.actor_critic.loaded_teachers}")
        print(f"[DEBUG] 教师数量: {self.actor_critic.num_teachers}")
        
        batch_count = 0
        
        # 生成小批次数据
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            
            batch_count += 1
            
            # ==================== 学生网络前向传播 ====================
            self.actor_critic.act(obs_batch, hist_encoding=False)
            student_actions = self.actor_critic.action_mean  # 使用确定性动作进行蒸馏
            
            # 调试信息：检查学生动作
            # if batch_count == 1:  # 只在第一个batch打印
            # 将第187行改为：
            sample_actions = student_actions[0].detach().cpu().numpy()
            formatted_actions = ", ".join([f"{x:.4f}" for x in sample_actions])
            print(f"[DEBUG] 学生[{0}] 完整动作: [{formatted_actions}]")
                # print(f"[DEBUG] 最终学生动作输出: [{student_actions.min().item():.4f}, {student_actions.max().item():.4f}]")
                # print(f"[DEBUG] 学生动作均值: {student_actions.mean().item():.4f}")
                # print(f"[DEBUG] 学生动作范围: [{student_actions.min().item():.4f}, {student_actions.max().item():.4f}]")
                # print(f"[DEBUG] 学生动作标准差: {student_actions.std().item():.4f}")
                # print(f"[DEBUG] 观测batch形状: {obs_batch.shape}")
                # # 检查学生动作是否异常
                # if torch.isnan(student_actions).any() or torch.isinf(student_actions).any():
                #     print(f"[ERROR] 学生动作包含NaN或Inf！")
                # if student_actions.abs().max() > 100.0:
                #     print(f"[WARNING] 学生动作异常大！")
            


            ###############################################################################
            #############测试相同obs下教师网络输出与原本的教师模型输出是否一致#################
            # "1、随机化obs，看教师网络输出与原本的教师模型输出是否一致"
            # import numpy as np

            # obs = np.array([
            #     [ 8.0546727e-03, -3.6421635e-03, -7.0213368e-03,  5.2526465e-04,
            #     -2.0166329e-04,  0.0000000e+00,  4.5037613e-04,  4.6616539e-04,
            #     0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
            #     0.0000000e+00,  3.0242365e-03,  2.2277385e-03, -1.2709423e-03,
            #     -6.6676140e-03, -2.7855635e-03, -4.8902232e-02, -1.2605737e-03,
            #     -2.5209486e-03, -3.5682821e-04,  8.0593824e-03,  2.0075887e-03,
            #     -4.8760224e-02,  9.3834549e-03,  5.2104406e-03, -2.1733518e-03,
            #     -1.4179296e-02, -5.5518586e-02, -4.9949858e-01, -2.8842248e-03,
            #     -7.0606768e-03,  1.0877823e-03,  2.6411880e-02, -4.9988132e-02,
            #     -4.9874470e-01,  1.9601990e-02, -1.3435342e-02, -1.0298284e-01,
            #     -4.9730211e-02,  7.6743625e-03,  2.1113213e-02, -8.6830556e-04,
            #     2.9343305e-02,  1.3911884e-02,  5.6031618e-02,  8.9739569e-02,
            #     2.4020143e-02, -5.0000000e-01, -5.0000000e-01,  6.7131007e-01,
            #     6.7131007e-01,  6.6631007e-01,  6.7631006e-01,  6.8131006e-01,
            #     6.7631006e-01,  6.7131007e-01,  6.7631006e-01,  6.7631006e-01,
            #     6.7631006e-01,  6.7131007e-01,  6.6131008e-01,  6.5631008e-01,
            #     6.6131008e-01,  6.6131008e-01,  6.5631008e-01,  6.5131009e-01,
            #     6.5631008e-01,  6.5631008e-01,  6.5631008e-01,  6.5131009e-01,
            #     6.6631007e-01,  6.4131004e-01,  6.4131004e-01,  6.3631004e-01,
            #     6.4131004e-01,  6.4131004e-01,  6.4131004e-01,  6.4631009e-01,
            #     6.4131004e-01,  6.3631004e-01,  6.4131004e-01,  6.3631004e-01,
            #     6.1631006e-01,  6.1131006e-01,  6.2131006e-01,  6.2131006e-01,
            #     6.2131006e-01,  6.2131006e-01,  6.1631006e-01,  6.2131006e-01,
            #     6.1631006e-01,  6.2131006e-01,  6.2131006e-01,  5.9131002e-01,
            #     6.0131007e-01,  6.0131007e-01,  6.0131007e-01,  5.9131002e-01,
            #     5.9631008e-01,  6.0131007e-01,  6.0631007e-01,  5.9131002e-01,
            #     5.9631008e-01,  5.9631008e-01,  5.8631003e-01,  5.8631003e-01,
            #     5.8631003e-01,  5.8631003e-01,  5.8131003e-01,  5.7631004e-01,
            #     5.8131003e-01,  5.8631003e-01,  5.7631004e-01,  5.8131003e-01,
            #     5.7631004e-01,  5.5631006e-01,  5.6631005e-01,  5.6631005e-01,
            #     5.7131004e-01,  5.6631005e-01,  5.6131005e-01,  5.5631006e-01,
            #     5.6631005e-01,  5.6631005e-01,  5.6131005e-01,  5.6631005e-01,
            #     5.3631008e-01,  5.5131006e-01,  5.4131007e-01,  5.4131007e-01,
            #     5.3631008e-01,  5.4131007e-01,  5.4131007e-01,  5.4631007e-01,
            #     5.4631007e-01,  5.5131006e-01,  5.5131006e-01,  5.2631009e-01,
            #     5.3631008e-01,  5.3631008e-01,  5.2631009e-01,  5.2631009e-01,
            #     5.3131008e-01,  5.2131009e-01,  5.3131008e-01,  5.3131008e-01,
            #     5.3131008e-01,  5.3631008e-01,  5.0631005e-01,  5.0631005e-01,
            #     5.1131004e-01,  5.0131005e-01,  5.0631005e-01,  5.0631005e-01,
            #     5.0631005e-01,  5.1131004e-01,  5.0631005e-01,  5.0131005e-01,
            #     5.0631005e-01,  4.8631006e-01,  4.9131006e-01,  4.9131006e-01,
            #     4.8631006e-01,  4.8631006e-01,  4.9631006e-01,  4.8631006e-01,
            #     4.9131006e-01,  4.8131007e-01,  4.8131007e-01,  4.9631006e-01,
            #     4.7131005e-01,  4.7131005e-01,  4.8131007e-01,  4.8131007e-01,
            #     4.8131007e-01,  4.8131007e-01,  4.7131005e-01,  4.7631007e-01,
            #     4.7631007e-01,  4.7131005e-01,  4.7631007e-01, -2.5763062e-05,
            #     -7.2046272e-03, -4.9383715e-01, -0.0000000e+00, -0.0000000e+00,
            #     -0.0000000e+00, -0.0000000e+00, -0.0000000e+00, -0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     8.0000001e-01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00, -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,
            #     1.5942507e-12, -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  9.3002582e-01,
            #     1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00, -5.0000000e-01, -5.0000000e-01,
            #     -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,  1.5942507e-12,
            #     -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00, -5.0000000e-01, -5.0000000e-01, -4.8428811e-10,
            #     5.9604610e-10, -1.9208597e-10,  1.5942507e-12, -5.1849698e-09,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  9.3002582e-01,  1.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     -5.0000000e-01, -5.0000000e-01, -4.8428811e-10,  5.9604610e-10,
            #     -1.9208597e-10,  1.5942507e-12, -5.1849698e-09,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     9.3002582e-01,  1.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -5.0000000e-01,
            #     -5.0000000e-01, -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,
            #     1.5942507e-12, -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  9.3002582e-01,
            #     1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00, -5.0000000e-01, -5.0000000e-01,
            #     -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,  1.5942507e-12,
            #     -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00, -5.0000000e-01, -5.0000000e-01, -4.8428811e-10,
            #     5.9604610e-10, -1.9208597e-10,  1.5942507e-12, -5.1849698e-09,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  9.3002582e-01,  1.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -5.0000000e-01,
            #     -5.0000000e-01, -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,
            #     1.5942507e-12, -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  9.3002582e-01,
            #     1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00, -5.0000000e-01, -5.0000000e-01,
            #     -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,  1.5942507e-12,
            #     -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
            #     0.0000000e+00, -5.0000000e-01, -5.0000000e-01]
            # ])

            # # print(f"obs shape: {obs.shape}")  # 输出: (1, 731)
            # obs_731 = np.zeros((1, 731))
            # obs_731[0, :679] = obs[0]
            
            # batch_size = 3072
            # obs_batch = np.tile(obs_731, (batch_size, 1))
            # obs_batch = torch.from_numpy(obs_batch).float().to("cuda")

            # print(f"obs_batch shape: {obs_batch.shape}")

########################################################################################
########################################################################################
            # ==================== 教师网络推理 ====================
            # 从环境信息中提取地形ID，每个教师对应一种地形
            terrain_ids = self._extract_terrain_ids(obs_batch)
            
            # 调试信息：检查地形ID分布
            if batch_count == 1:
                unique_terrains, counts = torch.unique(terrain_ids, return_counts=True)
                print(f"[DEBUG] 地形ID分布: {dict(zip(unique_terrains.cpu().numpy(), counts.cpu().numpy()))}")
            
            # ==================== 获取教师动作（通过真实环境交互）====================            
            # 1. 获取教师动作（与play.py和学生一致）
            teacher_actions = self.actor_critic.get_teacher_actions(
                obs_batch, 
                terrain_ids=terrain_ids,  # 使用地形ID选择对应教师
                hist_encoding=True,  # 与play.py保持一致
                env=self.env  # 传入环境实例进行真实处理
            )
            
           
            # if batch_count == 1:
            sample_actions = teacher_actions[0].detach().cpu().numpy()
            formatted_actions = ", ".join([f"{x:.4f}" for x in sample_actions])
            print(f"[DEBUG] 教师[{0}] 完整动作: [{formatted_actions}]")
                # print(f"[DEBUG] 最终教师动作输出: [{teacher_processed_actions.min().item():.4f}, {teacher_processed_actions.max().item():.4f}]")
                # print(f"[DEBUG] 教师动作均值: {teacher_processed_actions.mean().item():.4f}")
                # print(f"[DEBUG] 教师动作范围: [{teacher_processed_actions.min().item():.4f}, {teacher_processed_actions.max().item():.4f}]")
                # print(f"[DEBUG] 教师动作标准差: {teacher_processed_actions.std().item():.4f}")
                # print(f"[DEBUG] 观测batch形状: {obs_batch.shape}")
                # 检查教师动作是否异常
                # if torch.isnan(teacher_processed_actions).any() or torch.isinf(teacher_processed_actions).any():
                #     print(f"[ERROR] 教师动作包含NaN或Inf！")
                # if teacher_processed_actions.abs().max() > 100.0:
                #     print(f"[WARNING] 教师动作异常大！")
                # print(f"[DEBUG] 教师动作已通过env.step()处理:")
                # print(f"[DEBUG] 教师处理后动作: [{teacher_processed_actions.min().item():.4f}, {teacher_processed_actions.max().item():.4f}]")
            
            # ==================== 计算蒸馏损失 ====================
            # 现在将学生和教师动作模拟真实环境进行处理
            student_processed = self.process_actions(student_actions)  # 学生动作通过process_actions处理
            teacher_processed = self.process_actions(teacher_actions)  # 教师动作通过process_actions处理
            sample_actions1 = teacher_processed[0].detach().cpu().numpy()
            sample_actions2 = student_processed[0].detach().cpu().numpy()
            formatted_actions_teacher = ", ".join([f"{x:.4f}" for x in sample_actions1])
            formatted_actions_student = ", ".join([f"{x:.4f}" for x in sample_actions2])
            print(f"[DEBUG] 教师 处理后动作: [{formatted_actions_teacher}]")
            # print(f"[DEBUG] 学生 处理前动作: [{student_actions}]")
            print(f"[DEBUG] 学生 处理后动作: [{formatted_actions_student}]")
            # assert False
                        
            # 1. 行为克隆损失 - 学生模仿当前地形对应教师的动作（比较真实环境处理后的动作）
            behavior_cloning_loss = F.mse_loss(student_processed, teacher_processed)
            
            # 2. 地形自适应权重 - 根据地形类型调整损失权重
            if self.terrain_adaptive_weights and terrain_ids is not None:
                adaptive_weights = self.actor_critic.get_terrain_adaptive_weights(
                    terrain_ids, 
                    temperature=self.weight_temperature
                )
                # 加权行为克隆损失（使用处理后的动作）
                weighted_bc_loss = self._compute_weighted_loss(
                    student_processed, 
                    teacher_processed, 
                    adaptive_weights
                )
                behavior_cloning_loss = weighted_bc_loss
            
            # 3. 多样性损失 - 鼓励学生在不同地形下学习不同的专门化行为（使用处理后的动作）
            diversity_loss = self._compute_diversity_loss(student_processed, terrain_ids)
            
            # 4. 总蒸馏损失
            distillation_loss = (
                self.behavior_cloning_coef * behavior_cloning_loss +
                self.diversity_loss_coef * diversity_loss
            )
            
            total_loss = self.distillation_loss_coef * distillation_loss
            
            # ==================== 梯度更新 ====================
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # 记录损失
            mean_distillation_loss += distillation_loss.item()
            mean_behavior_cloning_loss += behavior_cloning_loss.item()
            mean_diversity_loss += diversity_loss.item()
            mean_total_loss += total_loss.item()
        
        # 计算平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_distillation_loss /= num_updates
        mean_behavior_cloning_loss /= num_updates
        mean_diversity_loss /= num_updates
        mean_total_loss /= num_updates
        
        # 清空存储
        self.storage.clear()
        self.update_counter()
        
        # 返回损失字典（兼容rsl_rl_old格式）
        return (
            mean_total_loss,           # 主损失
            mean_distillation_loss,    # 蒸馏损失
            mean_behavior_cloning_loss, # 行为克隆损失
            mean_diversity_loss,       # 多样性损失
            0,                         # 占位符
            0,                         # 占位符
            0                          # 占位符
        )
    
    def _extract_terrain_ids(self, obs_batch):
        """从观测中提取地形ID，正确映射到对应的教师
        
        当前配置：只有1个教师（hurdle地形），所有样本都使用教师0
        """
        batch_size = obs_batch.shape[0]
        num_teachers = self.actor_critic.num_teachers
        
        # 初始化terrain_ids
        terrain_ids = torch.zeros(batch_size, dtype=torch.long, device=obs_batch.device)
        
        if num_teachers == 1:
            # 单教师模式：所有样本都使用教师0
            print(f"[DEBUG] 单教师模式：所有{batch_size}个样本都使用教师0")
        else:
            # 多教师模式：将batch均匀分配给所有教师
            batch_per_teacher = batch_size // num_teachers
            remainder = batch_size % num_teachers
            
            start_idx = 0
            for teacher_id in range(num_teachers):
                # 计算当前教师负责的样本数量
                current_batch_size = batch_per_teacher + (1 if teacher_id < remainder else 0)
                end_idx = start_idx + current_batch_size
                
                # 分配给当前教师
                if end_idx > start_idx:
                    terrain_ids[start_idx:end_idx] = teacher_id
                start_idx = end_idx
        
        return terrain_ids
    
    def _compute_weighted_loss(self, student_actions, teacher_actions, weights):
        """计算加权损失"""
        # student_actions: [batch_size, action_dim]
        # teacher_actions: [batch_size, action_dim] 
        # weights: [batch_size, num_teachers]
        
        # 简化版本：使用权重的最大值作为样本权重
        sample_weights = torch.max(weights, dim=1)[0]  # [batch_size]
        
        # 计算每个样本的损失
        sample_losses = F.mse_loss(student_actions, teacher_actions, reduction='none').mean(dim=1)  # [batch_size]
        
        # 加权平均
        weighted_loss = (sample_losses * sample_weights).mean()
        
        return weighted_loss
    
    def _compute_diversity_loss(self, student_actions, terrain_ids):
        """计算多样性损失 - 鼓励不同地形下的不同行为"""
        if terrain_ids is None:
            return torch.tensor(0.0, device=student_actions.device)
        
        batch_size = student_actions.shape[0]
        
        # 计算同一地形内动作的方差（希望较小）
        diversity_loss = 0
        unique_terrain_ids = torch.unique(terrain_ids)
        
        for terrain_id in unique_terrain_ids:
            mask = (terrain_ids == terrain_id)
            if mask.sum() > 1:  # 需要至少2个样本计算方差
                terrain_actions = student_actions[mask]
                # 计算该地形内动作的一致性
                action_mean = terrain_actions.mean(dim=0, keepdim=True)
                consistency_loss = F.mse_loss(terrain_actions, action_mean.expand_as(terrain_actions))
                diversity_loss += consistency_loss
        
        if len(unique_terrain_ids) > 0:
            diversity_loss /= len(unique_terrain_ids)
        
        return diversity_loss
    
    def update_counter(self):
        """更新计数器"""
        self.counter += 1


# 导出类
__all__ = ["MultiTeacherDistillation"]