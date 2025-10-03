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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):
        """
        强化学习训练器初始化
        
        Args:
            env: 向量化环境，包含多个并行环境实例
            train_cfg: 训练配置字典，包含runner、algorithm、policy等配置
            log_dir: 日志保存目录
            init_wandb: 是否初始化wandb日志记录
            device: 计算设备（'cpu'或'cuda'）
        """

        # ========== 步骤1：解析训练配置 ==========
        # 从训练配置字典中提取各个组件的配置
        self.cfg = train_cfg["runner"]              # 训练器配置（学习率、更新频率等）
        self.alg_cfg = train_cfg["algorithm"]       # 算法配置（PPO超参数：clip范围、熵系数等）
        self.policy_cfg = train_cfg["policy"]       # 策略网络配置（隐藏层维度、激活函数等）
        self.estimator_cfg = train_cfg["estimator"] # 状态估计器配置（用于估计特权信息）
        self.depth_encoder_cfg = train_cfg["depth_encoder"]  # 深度编码器配置（视觉处理）
        
        # 设置计算设备和环境引用
        self.device = device
        self.env = env

        # ========== 步骤2：创建Actor-Critic网络 ==========
        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        
        # 创建RMA（Robust Motion Adaptation）风格的Actor-Critic网络
        # RMA架构特点：分离本体感受观测和特权信息，提高sim-to-real迁移能力
        actor_critic: ActorCriticRMA = ActorCriticRMA(
            self.env.cfg.env.n_proprio,        # 本体感受观测维度（IMU、关节状态等）
            self.env.cfg.env.n_scan,           # 扫描观测维度（高度图、激光雷达等）
            self.env.num_obs,                  # 总观测维度（所有观测的总和）
            self.env.cfg.env.n_priv_latent,    # 潜在特权信息维度（物理参数：质量、摩擦等）
            self.env.cfg.env.n_priv,           # 显式特权信息维度（线速度等可估计信息）
            self.env.cfg.env.history_len,      # 历史观测长度（时序信息窗口大小）
            self.env.num_actions,              # 动作维度（关节控制命令数）
            **self.policy_cfg                  # 其他策略网络配置（MLP层数、激活函数等）
        ).to(self.device)
        
        # ========== 步骤3：创建状态估计器 ==========
        # Estimator: 从本体感受观测估计特权信息（如线速度）
        # 训练时：使用真实特权信息监督学习
        # 部署时：替代仿真中的特权信息，实现sim-to-real迁移
        estimator = Estimator(
            input_dim=env.cfg.env.n_proprio,   # 输入：本体感受观测（IMU、关节等）
            output_dim=env.cfg.env.n_priv,     # 输出：估计的特权信息（线速度等）
            hidden_dims=self.estimator_cfg["hidden_dims"]  # 隐藏层维度列表 [256, 128, 64]
        ).to(self.device)
        
        # ========== 步骤4：创建深度编码器（可选）==========
        # 深度编码器用于处理深度相机图像，提供视觉感知能力
        self.if_depth = self.depth_encoder_cfg["if_depth"]  # 是否使用深度视觉
        
        if self.if_depth:
            # 创建深度图像处理的主干网络
            # DepthOnlyFCBackbone58x87: 处理58x87像素深度图像的全连接网络
            # 58x87是深度图像降采样后的分辨率，平衡计算效率和信息保留
            depth_backbone = DepthOnlyFCBackbone58x87(
                env.cfg.env.n_proprio,                    # 本体感受观测维度（与策略网络共享）
                self.policy_cfg["scan_encoder_dims"][-1], # 扫描编码器输出维度（特征维度）
                self.depth_encoder_cfg["hidden_dims"],    # 隐藏层维度配置
                                                    )
            
            # 创建循环深度编码器，处理时序深度图像
            # RecurrentDepthBackbone: 包含LSTM/GRU的循环网络
            # 处理深度图像序列，提取时空特征
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
            
            # 为深度编码器创建独立的Actor网络
            # 深拷贝原始actor，用于处理视觉输入的动作生成
            depth_actor = deepcopy(actor_critic.actor)
        else:
            # 不使用深度编码器时设为None
            depth_encoder = None
            depth_actor = None
            
        # 注释掉的代码：深度编码器的独立训练组件
        # self.depth_encoder = depth_encoder
        # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
        # self.depth_encoder_paras = self.depth_encoder_cfg
        # self.depth_encoder_criterion = nn.MSELoss()
        
        # ========== 步骤5：创建强化学习算法 ==========
        # 使用反射机制动态创建算法类
        # eval()将字符串"PPO"转换为PPO类对象
        alg_class = eval(self.cfg["algorithm_class_name"])  # 通常是 "PPO"
        
        # 创建PPO算法实例，整合所有网络组件
        self.alg: PPO = alg_class(
            actor_critic,                      # Actor-Critic主网络
            estimator, self.estimator_cfg,     # 状态估计器及其配置
            depth_encoder, self.depth_encoder_cfg, depth_actor,  # 深度编码器相关（可选）
            device=self.device,                # 计算设备
            **self.alg_cfg                     # PPO算法参数（学习率、clip范围等）
        )
        
        # ========== 步骤6：设置训练参数 ==========
        # 每个环境收集多少步数据后进行一次策略更新
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 例如：24步
        
        # 模型保存间隔（每多少次迭代保存一次）
        self.save_interval = self.cfg["save_interval"]  # 例如：50次迭代
        
        # DAgger（Dataset Aggregation）更新频率
        # DAgger用于改进状态估计器，通过收集更多数据提高估计精度
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]  # 例如：每10次更新

        # ========== 步骤7：初始化经验缓冲区 ==========
        # 为PPO算法初始化rollout存储缓冲区
        # 用于存储环境交互数据（观测、动作、奖励等）
        self.alg.init_storage(
            self.env.num_envs,                    # 并行环境数量（例如：4096）
            self.num_steps_per_env,               # 每个环境的步数（例如：24）
            [self.env.num_obs],                   # 观测维度列表
            [self.env.num_privileged_obs],        # 特权观测维度列表
            [self.env.num_actions],               # 动作维度列表
        )

        # ========== 步骤8：选择学习模式 ==========
        # 根据是否使用深度编码器选择不同的学习函数
        # learn_RL: 纯强化学习模式（只使用本体感受和高度图）
        # learn_vision: 视觉强化学习模式（额外使用深度相机）
        self.learn = self.learn_RL if not self.if_depth else self.learn_vision
            
        # ========== 步骤9：初始化日志记录 ==========
        self.log_dir = log_dir                    # 日志保存目录路径
        self.writer = None                        # TensorBoard写入器（稍后初始化）
        self.tot_timesteps = 0                    # 总训练步数计数器
        self.tot_time = 0                         # 总训练时间计数器（秒）
        self.current_learning_iteration = 0       # 当前学习迭代次数
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        纯强化学习训练函数（不使用视觉）
        
        Args:
            num_learning_iterations: 训练迭代次数
            init_at_random_ep_len: 是否随机初始化episode长度（用于课程学习）
        """
        
        # ========== 步骤1：初始化损失记录变量 ==========
        mean_value_loss = 0.            # 价值函数损失（Critic网络）
        mean_surrogate_loss = 0.        # 代理损失（PPO的策略损失）
        mean_estimator_loss = 0.        # 状态估计器损失
        mean_disc_loss = 0.             # 判别器损失（用于域适应）
        mean_disc_acc = 0.              # 判别器准确率
        mean_hist_latent_loss = 0.      # 历史潜在编码损失（DAgger）
        mean_priv_reg_loss = 0.         # 特权信息正则化损失
        priv_reg_coef = 0.              # 特权正则化系数
        entropy_coef = 0.               # 熵正则化系数
        
        # TensorBoard写入器初始化（已注释，使用wandb代替）
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        # ========== 步骤2：随机初始化episode长度（可选）==========
        # 用于课程学习，让机器人从不同的episode阶段开始训练
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, 
                high=int(self.env.max_episode_length)
            )
        
        # ========== 步骤3：获取初始观测 ==========
        obs = self.env.get_observations()                    # 获取观测（本体感受+高度图等）
        privileged_obs = self.env.get_privileged_observations()  # 获取特权观测（仿真中可得的真实信息）
        # Critic使用特权观测（如果有），Actor使用普通观测
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        # 初始化额外信息字典
        infos = {}
        # 深度信息：纯RL模式下为None（self.if_depth = False）
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        
        # 设置网络为训练模式（启用dropout等）
        self.alg.actor_critic.train()

        # ========== 步骤4：初始化日志记录缓冲区 ==========
        ep_infos = []                                          # episode信息列表
        rewbuffer = deque(maxlen=100)                         # 奖励缓冲区（最近100个episode）
        rew_explr_buffer = deque(maxlen=100)                  # 探索奖励缓冲区
        rew_entropy_buffer = deque(maxlen=100)                # 熵奖励缓冲区
        lenbuffer = deque(maxlen=100)                         # episode长度缓冲区
        
        # 当前episode的累计值（每个环境一个）
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)        # 总奖励累计
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 探索奖励累计
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) # 熵奖励累计
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)    # episode长度

        # ========== 步骤5：设置训练迭代范围 ==========
        tot_iter = self.current_learning_iteration + num_learning_iterations  # 总迭代次数
        self.start_learning_iteration = copy(self.current_learning_iteration)  # 记录起始迭代

        # ========== 步骤6：主训练循环 ==========
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # 判断是否需要更新历史编码器（DAgger）
            # 每dagger_update_freq次迭代更新一次
            hist_encoding = it % self.dagger_update_freq == 0

            # ========== 步骤6.1：收集经验（Rollout）==========
            with torch.inference_mode():  # 推理模式，不计算梯度
                for i in range(self.num_steps_per_env):  # 收集num_steps_per_env步数据
                    # 根据当前观测生成动作
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    
                    # 环境步进：执行动作，获取新观测、奖励、终止信号
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    # 注意：如果done=True，obs已经是重置后的新观测
                    
                    # 更新critic观测
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    
                    # 将数据转移到GPU
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device), 
                        critic_obs.to(self.device), 
                        rewards.to(self.device), 
                        dones.to(self.device)
                    )
                    
                    # 处理环境步骤，存储到经验缓冲区
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    # ========== 日志记录 ==========
                    if self.log_dir is not None:
                        # 记录episode信息
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        
                        # 累计奖励和长度
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0  # 探索奖励（当前未使用）
                        cur_reward_entropy_sum += 0  # 熵奖励（当前未使用）
                        cur_episode_length += 1
                        
                        # 处理完成的episode
                        new_ids = (dones > 0).nonzero(as_tuple=False)  # 找出完成的环境
                        
                        # 将完成的episode数据加入缓冲区
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        # 重置完成的环境的累计值
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start  # 数据收集时间

                # ========== 步骤6.2：计算回报（Returns）==========
                start = stop
                # 使用GAE（Generalized Advantage Estimation）计算优势函数
                self.alg.compute_returns(critic_obs)
            
            # ========== 步骤6.3：策略更新 ==========
            # PPO更新：包括Actor、Critic、Estimator等所有网络
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            
            # DAgger更新：更新历史编码器（如果需要）
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start  # 学习时间
            
            # ========== 步骤6.4：日志记录和模型保存 ==========
            if self.log_dir is not None:
                self.log(locals())  # 记录所有局部变量到日志
            
            # 模型保存策略：
            # - 前2500次迭代：每save_interval保存一次
            # - 2500-5000次：每2*save_interval保存一次  
            # - 5000次后：每5*save_interval保存一次
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            
            # 清空episode信息，准备下一次迭代
            ep_infos.clear()
        
        # ========== 步骤7：训练结束，保存最终模型 ==========
        self.current_learning_iteration += num_learning_iterations  # 更新迭代计数
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        视觉强化学习训练函数（使用深度相机）
        使用教师-学生架构进行知识蒸馏
        
        Args:
            num_learning_iterations: 训练迭代次数
            init_at_random_ep_len: 是否随机初始化episode长度（未使用）
        """
        
        # ========== 步骤1：初始化训练参数 ==========
        tot_iter = self.current_learning_iteration + num_learning_iterations  # 总迭代次数
        self.start_learning_iteration = copy(self.current_learning_iteration)  # 记录起始迭代

        # ========== 步骤2：初始化日志记录缓冲区 ==========
        ep_infos = []                          # episode信息列表
        rewbuffer = deque(maxlen=100)         # 奖励缓冲区（最近100个episode）
        lenbuffer = deque(maxlen=100)         # episode长度缓冲区
        # 当前episode的累计值（每个环境一个）
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)      # 总奖励累计
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # episode长度

        # ========== 步骤3：获取初始观测和深度图像 ==========
        obs = self.env.get_observations()     # 获取观测（包含本体感受、高度图等）
        infos = {}
        # 获取最新的深度图像（深度缓冲区的最后一帧）
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        # 初始化朝向误差有效标志（用于筛选有效的朝向预测）
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        
        # 设置深度编码器和深度Actor为训练模式
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        # ========== 步骤4：设置预训练迭代次数 ==========
        num_pretrain_iter = 0  # 预训练阶段使用教师策略的迭代次数（当前设为0，直接使用学生策略）
        
        # ========== 步骤5：主训练循环 ==========
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # 初始化数据缓冲区（用于批量更新）
            depth_latent_buffer = []        # 深度编码器输出的潜在表示
            scandots_latent_buffer = []     # 高度图编码器的潜在表示（教师网络）
            actions_teacher_buffer = []     # 教师策略的动作
            actions_student_buffer = []     # 学生策略的动作
            yaw_buffer_student = []         # 学生网络预测的朝向
            yaw_buffer_teacher = []         # 教师网络使用的真实朝向
            delta_yaw_ok_buffer = []        # 朝向预测有效性统计
            
            # ========== 步骤5.1：收集经验 ==========
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                
                # ========== 处理深度图像 ==========
                if infos["depth"] != None:
                    # 获取教师网络的高度图潜在表示（用于对比学习）
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    
                    # 准备深度编码器的输入
                    # 提取本体感受观测，并将朝向信息置零（让网络自己预测）
                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0  # 清零朝向信息（delta_yaw）
                    
                    # 深度编码器前向传播：输入深度图像和本体感受，输出潜在表示和朝向预测
                    depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)
                    # 注意：clone()很重要，避免原地操作
                    
                    # 分离潜在表示和朝向预测
                    depth_latent = depth_latent_and_yaw[:, :-2]  # 深度特征（用于替代高度图）
                    yaw = 1.5 * depth_latent_and_yaw[:, -2:]    # 朝向预测（乘以1.5是缩放因子）
                    
                    # 保存到缓冲区
                    depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(yaw)
                    yaw_buffer_teacher.append(obs[:, 6:8])  # 真实朝向（从观测中提取）
                
                # ========== 生成教师动作（使用特权信息）==========
                with torch.no_grad():
                    # 教师策略使用完整观测（包括真实朝向和高度图）
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                # ========== 生成学生动作（使用视觉信息）==========
                obs_student = obs.clone()
                # 只在朝向预测有效的环境中替换朝向信息
                # obs_student[:, 6:8] = yaw.detach()  # 完全替换（已注释）
                obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]  # 选择性替换
                
                # 统计朝向预测有效的比例
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
                
                # 学生策略使用深度特征替代高度图
                actions_student = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # ========== 环境步进 ==========
                # 在预训练阶段使用教师动作，之后使用学生动作
                if it < num_pretrain_iter:
                    # 预训练：使用教师策略收集数据
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_teacher.detach())
                else:
                    # 正式训练：使用学生策略收集数据
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())
                # 注意：如果done=True，obs已经是重置后的新观测
                
                # 更新观测数据
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = (
                    obs.to(self.device), 
                    critic_obs.to(self.device), 
                    rewards.to(self.device), 
                    dones.to(self.device)
                )

                # ========== 日志记录 ==========
                if self.log_dir is not None:
                    # 记录episode信息
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                    
                    # 累计奖励和长度
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                    
                    # 处理完成的episode
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    
                    # 重置完成的环境
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
            stop = time.time()
            collection_time = stop - start  # 数据收集时间
            start = stop

            # ========== 步骤5.2：网络更新 ==========
            
            # 计算朝向预测有效率
            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            
            # 将列表缓冲区转换为张量
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)  # [batch_size, latent_dim]
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)        # [batch_size, latent_dim]
            
            # 深度编码器损失（当前未启用）
            depth_encoder_loss = 0
            # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

            # 准备动作和朝向数据
            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)  # [batch_size, action_dim]
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)  # [batch_size, action_dim]
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)         # [batch_size, 2]
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)         # [batch_size, 2]
            
            # 更新深度Actor（知识蒸馏）
            # 损失包括：动作模仿损失 + 朝向预测损失
            depth_actor_loss, yaw_loss = self.alg.update_depth_actor(
                actions_student_buffer,   # 学生动作
                actions_teacher_buffer,   # 教师动作（监督信号）
                yaw_buffer_student,       # 学生朝向预测
                yaw_buffer_teacher        # 真实朝向（监督信号）
            )
            
            # 联合更新（已注释，可选方案）
            # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(
            #     depth_latent_buffer, scandots_latent_buffer, 
            #     actions_student_buffer, actions_teacher_buffer
            # )
            
            stop = time.time()
            learn_time = stop - start  # 学习时间

            # ========== 步骤5.3：清理和日志 ==========

            # 分离循环网络的隐藏状态，避免梯度累积
            self.alg.depth_encoder.detach_hidden_states()

            # 记录训练日志
            if self.log_dir is not None:
                self.log_vision(locals())  # 记录视觉训练特有的指标
            
            # 模型保存策略（与learn_RL相同）
            if (it-self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
               (it-self.start_learning_iteration < 5000 and it % (2*self.save_interval) == 0) or \
               (it-self.start_learning_iteration >= 5000 and it % (5*self.save_interval) == 0):
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            
            # 清空episode信息
            ep_infos.clear()
    
    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    
        # 添加关键KPI监控（策略监控指标）
        if hasattr(self.env, 'total_times') and hasattr(self.env, 'success_times') and hasattr(self.env, 'complete_times'):
            if self.env.total_times > 0:
                success_rate = self.env.success_times / self.env.total_times
                completion_rate = self.env.complete_times / self.env.total_times
                wandb_dict['Episode_rew/success_rate'] = success_rate
                wandb_dict['Episode_rew/completion_rate'] = completion_rate
                wandb_dict['Episode_rew/terrain_level'] = torch.mean(self.env.terrain_levels.float()) if hasattr(self.env, 'terrain_levels') else 0
    
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder
    
    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
