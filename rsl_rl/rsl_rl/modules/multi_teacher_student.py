# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Multi-Teacher Student Network for H1 Robot - Based on rsl_rl_old framework
# 多教师学生网络 - H1机器人版本，基于rsl_rl_old框架

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
import warnings
from typing import Dict, List, Optional, Union

from .actor_critic import ActorCriticRMA, Actor, get_activation


class MultiTeacherStudent(nn.Module):
    """多教师学生网络 - 基于rsl_rl_old的ActorCriticRMA架构
    
    该网络包含：
    1. 学生策略网络：执行实际动作
    2. 多个教师策略网络：提供监督信号
    3. 地形自适应路由：根据地形选择教师
    """
    
    is_recurrent = False
    
    def __init__(
        self,
        num_prop: int,
        num_scan: int, 
        num_critic_obs: int,
        num_priv_latent: int,
        num_priv_explicit: int,
        num_hist: int,
        num_actions: int,
        num_teachers: int = 6,
        teacher_model_paths: Optional[List[str]] = None,
        scan_encoder_dims: List[int] = [256, 256, 256],
        actor_hidden_dims: List[int] = [256, 256, 256], 
        critic_hidden_dims: List[int] = [256, 256, 256],
        teacher_hidden_dims: List[int] = [256, 256, 256],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        **kwargs
    ):
        """初始化多教师学生网络
        
        Args:
            num_prop: 本体感受观测维度
            num_scan: 高度扫描观测维度  
            num_critic_obs: Critic观测维度
            num_priv_latent: 潜在特权信息维度
            num_priv_explicit: 显式特权信息维度
            num_hist: 历史观测长度
            num_actions: 动作维度
            num_teachers: 教师数量
            teacher_model_paths: 教师模型路径列表
            其他参数: 网络结构配置参数
        """
        super().__init__()
        
        if kwargs:
            print("MultiTeacherStudent.__init__ got unexpected arguments: " + str([key for key in kwargs.keys()]))
        
        # 保存配置参数
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_critic_obs = num_critic_obs
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_teachers = num_teachers
        self.teacher_model_paths = teacher_model_paths or []
        print(f"[DEBUG] 教师模型路径: {self.teacher_model_paths}")
        
        # 激活函数
        activation_fn = get_activation(activation)
        
        # ==================== 创建学生网络 ====================
        # 学生Actor网络
        self.student_actor = Actor(
            num_prop=num_prop,
            num_scan=num_scan,
            num_actions=num_actions,
            scan_encoder_dims=scan_encoder_dims,
            actor_hidden_dims=actor_hidden_dims,
            priv_encoder_dims=kwargs.get('priv_encoder_dims', []),
            num_priv_latent=num_priv_latent,
            num_priv_explicit=num_priv_explicit,
            num_hist=num_hist,
            activation=activation_fn,
            tanh_encoder_output=kwargs.get('tanh_encoder_output', False)
        )
        
        # 学生Critic网络
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.student_critic = nn.Sequential(*critic_layers)
        
        # 学生动作噪声
        self.student_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # ==================== 创建教师网络 ====================
        self.teacher_actors = nn.ModuleList()
        self.teacher_stds = nn.ParameterList()
        
        for i in range(num_teachers):
            # 关键修复：使用H1_2FixCfg的正确配置
            # 从搜索结果得知：H1_2FixCfg.env.n_priv = 3 + 3 + 3 = 9 (不是29!)
            # H1_2FixCfg.env.n_priv_latent = 4 + 1 + 12 + 12 = 29
            
            if self.teacher_model_paths and i < len(self.teacher_model_paths):
                try:
                    model_path = self.teacher_model_paths[i]
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # 从检查点读取网络结构以验证
                    if 'actor.actor_backbone.0.weight' in state_dict:
                        backbone_first_weight = state_dict['actor.actor_backbone.0.weight']
                        backbone_input_dim = backbone_first_weight.shape[1]  # 实际输入维度
                        print(f"[DEBUG] 教师{i} 检查点backbone输入维度: {backbone_input_dim}")
                        
                        # 使用H1的正确配置验证
                        # backbone_input = num_prop + scan_encoder_output + num_priv_explicit + priv_encoder_output
                        # = 51 + 32 + 9 + 20 = 112 ✓
                        expected_input = 51 + 32 + 9 + 20  # 112
                        print(f"[DEBUG] 教师{i} 期望输入维度: {expected_input} (51+32+9+20)")
                        print(f"[DEBUG] 教师{i} 维度匹配: {expected_input == backbone_input_dim}")
                    
                    # 分析priv_encoder结构（从检查点推断）
                    if 'actor.priv_encoder.0.weight' in state_dict and 'actor.priv_encoder.2.weight' in state_dict:
                        priv_first_weight = state_dict['actor.priv_encoder.0.weight']
                        priv_last_weight = state_dict['actor.priv_encoder.2.weight']
                        priv_input_dim = priv_first_weight.shape[1]  # priv_encoder输入
                        priv_hidden_dim = priv_first_weight.shape[0]  # priv_encoder隐藏层
                        priv_output_dim = priv_last_weight.shape[0]   # priv_encoder输出
                        
                        actual_priv_encoder_dims = [priv_hidden_dim, priv_output_dim]
                        actual_num_priv_latent = priv_input_dim
                        
                        print(f"[DEBUG] 教师{i} 检查点priv_encoder: {priv_input_dim}->{priv_hidden_dim}->{priv_output_dim}")
                    else:
                        # 使用默认值
                        actual_priv_encoder_dims = [64, 20]
                        actual_num_priv_latent = 29  # H1配置
                        
                except Exception as e:
                    print(f"[ERROR] 分析教师{i}网络结构失败: {e}")
                    actual_priv_encoder_dims = [64, 20]
                    actual_num_priv_latent = 29
            else:
                # 默认配置
                actual_priv_encoder_dims = [64, 20]
                actual_num_priv_latent = 29
            
            # 使用H1_2FixCfg的正确配置创建教师网络
            # 关键修复：传递原始activation字符串，让ActorCriticRMA内部处理激活函数转换
            teacher_actor_critic = ActorCriticRMA(
                num_prop=51,  # H1_2FixCfg.env.n_proprio
                num_scan=132,  # H1_2FixCfg.env.n_scan  
                num_critic_obs=num_critic_obs,
                num_priv_latent=actual_num_priv_latent,  # 29, H1_2FixCfg.env.n_priv_latent
                num_priv_explicit=9,  # 关键修复：H1_2FixCfg.env.n_priv = 3+3+3 = 9
                num_hist=10,  # H1_2FixCfg.env.history_len
                num_actions=num_actions,
                actor_hidden_dims=[512, 256, 128],  # 从检查点推断
                critic_hidden_dims=[512, 256, 128],  # 默认值
                scan_encoder_dims=[128, 64, 32],  # 从检查点推断
                activation=activation,  # 关键修复：传递原始字符串，避免重复转换
                init_noise_std=init_noise_std,
                priv_encoder_dims=actual_priv_encoder_dims,  # 从检查点推断
                tanh_encoder_output=False  # 关键：与play.py完全一致！
            )
            self.teacher_actors.append(teacher_actor_critic)
            
            print(f"[DEBUG] 教师{i} 最终网络配置:")
            print(f"  - num_priv_latent: {actual_num_priv_latent}")
            print(f"  - num_priv_explicit: 9 (H1配置)")
            print(f"  - priv_encoder_dims: {actual_priv_encoder_dims}")
            print(f"  - activation: {activation} (字符串)")
            print(f"  - tanh_encoder_output: False")
            
            # 教师动作噪声
            teacher_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.teacher_stds.append(teacher_std)
        
        # ==================== 观测标准化 ====================
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        
        # ==================== 加载教师模型 ====================
        self.loaded_teachers = False
        if teacher_model_paths:
            self.load_teacher_models(teacher_model_paths)
            # assert False
        
        # 禁用默认参数验证以加速
        Normal.set_default_validate_args = False
    
    def load_teacher_models(self, model_paths: List[str]):
        """加载教师模型权重"""
        if len(model_paths) != self.num_teachers:
            raise ValueError(f"教师模型路径数量({len(model_paths)})与教师数量({self.num_teachers})不匹配")
        
        loaded_count = 0
        for i, model_path in enumerate(model_paths):
            if not os.path.exists(model_path):
                warnings.warn(f"教师模型路径不存在: {model_path}")
                continue
                
            try:
                # 加载模型检查点
                print(f"[DEBUG] 正在加载教师模型 {i}: {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
                print(f"[DEBUG] 教师{i}模型键: {list(checkpoint.keys())}")
                
                # 提取actor_critic状态字典
                if 'ac_state_dict' in checkpoint:
                    ac_state_dict = checkpoint['ac_state_dict']
                    print(f"[DEBUG] 使用 'ac_state_dict' 键")
                elif 'model_state_dict' in checkpoint:
                    ac_state_dict = checkpoint['model_state_dict']
                    print(f"[DEBUG] 使用 'model_state_dict' 键")
                else:
                    ac_state_dict = checkpoint
                    print(f"[DEBUG] 直接使用checkpoint作为状态字典")
                
                print(f"[DEBUG] 状态字典键数量: {len(ac_state_dict)}")
                print(f"[DEBUG] 前10个键: {list(ac_state_dict.keys())[:10]}")
                
                # 手动加载匹配的参数（避免维度不匹配错误）
                print(f"[DEBUG] 手动加载教师{i} ActorCritic匹配的参数...")
                
                try:
                    # 尝试直接加载
                    missing_keys, unexpected_keys = self.teacher_actors[i].load_state_dict(
                        ac_state_dict, strict=False
                    )
                    print(f"[DEBUG] 教师{i} ActorCritic直接加载成功")
                except RuntimeError as e:
                    print(f"[DEBUG] 直接加载失败，使用手动匹配方式: {str(e)[:100]}...")
                    # 手动加载匹配的参数
                    model_dict = self.teacher_actors[i].state_dict()
                    matched_dict = {}
                    skipped_params = []
                    
                    for k, v in ac_state_dict.items():
                        if k in model_dict and model_dict[k].shape == v.shape:
                            matched_dict[k] = v
                        else:
                            skipped_params.append(k)
                    
                    missing_keys, unexpected_keys = self.teacher_actors[i].load_state_dict(matched_dict, strict=False)
                    print(f"[DEBUG] 手动加载了 {len(matched_dict)} 个匹配参数，跳过了 {len(skipped_params)} 个不匹配参数")
                    if skipped_params:
                        print(f"[DEBUG] 跳过的参数: {skipped_params[:3]}...")  # 只显示前3个
                
                print(f"[DEBUG] 教师{i} ActorCritic加载: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
                
                # 从加载的ActorCritic中获取std参数
                if hasattr(self.teacher_actors[i], 'std'):
                    teacher_std_value = self.teacher_actors[i].std.data.clone()
                    self.teacher_stds[i].data.copy_(teacher_std_value)
                    print(f"[INFO] 教师{i} std参数: max={teacher_std_value.max().item():.4f}, min={teacher_std_value.min().item():.4f}")
                else:
                    print(f"[WARNING] 教师{i} ActorCritic没有std属性")
                
                print(f"成功加载教师模型 {i}: {model_path}")
                if missing_keys:
                    print(f"教师{i}缺少键: {missing_keys}")
                if unexpected_keys:
                    print(f"教师{i}多余键: {unexpected_keys}")
                
                # 调试信息：检查教师ActorCritic参数
                total_params = sum(p.numel() for p in self.teacher_actors[i].parameters())
                print(f"[DEBUG] 教师{i} ActorCritic参数数量: {total_params}")
                
                # 测试教师ActorCritic是否能正常前向传播
                with torch.no_grad():
                    # 使用731维的测试观测（与环境观测维度匹配）
                    test_obs = torch.randn(1, 731, device='cpu')  # 先在CPU上创建
                    
                    # 应用与训练时相同的观测裁剪
                    test_obs_clipped = torch.clamp(test_obs, -100.0, 100.0)
                    
                    # 使用ActorCritic的act_inference方法（与play.py一致）
                    test_action = self.teacher_actors[i].act_inference(test_obs_clipped, hist_encoding=True, eval=False)
                    print(f"[DEBUG] 教师{i}测试动作范围: [{test_action.min().item():.4f}, {test_action.max().item():.4f}]")
                    
                    # 测试观测数值范围
                    print(f"[DEBUG] 测试观测范围: [{test_obs_clipped.min().item():.4f}, {test_obs_clipped.max().item():.4f}]")
                    print(f"[DEBUG] 测试观测均值: {test_obs_clipped.mean().item():.4f}, 标准差: {test_obs_clipped.std().item():.4f}")
                    
                    # 测试零观测
                    zero_obs = torch.zeros(1, 731, device='cpu')
                    zero_action = self.teacher_actors[i].act_inference(zero_obs, hist_encoding=True, eval=False)
                    print(f"[DEBUG] 教师{i}零观测动作范围: [{zero_action.min().item():.4f}, {zero_action.max().item():.4f}]")
                
                loaded_count += 1
                
            except Exception as e:
                warnings.warn(f"加载教师模型{i}失败: {model_path}, 错误: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.loaded_teachers = loaded_count > 0
        print(f"成功加载 {loaded_count}/{self.num_teachers} 个教师模型")
        
        # 冻结教师网络参数
        for teacher_actor in self.teacher_actors:
            for param in teacher_actor.parameters():
                param.requires_grad = False
        
        for teacher_std in self.teacher_stds:
            teacher_std.requires_grad = False
    
    def reset(self, dones=None):
        """重置网络状态（兼容rsl_rl_old接口）"""
        pass
    
    def forward(self):
        """前向传播（rsl_rl_old要求的接口，但不实际使用）"""
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """获取动作均值（兼容rsl_rl_old接口）"""
        return self.distribution.mean if self.distribution else None

    @property
    def action_std(self):
        """获取动作标准差（兼容rsl_rl_old接口）"""
        return self.distribution.stddev if self.distribution else None
    
    @property
    def entropy(self):
        """获取动作熵（兼容rsl_rl_old接口）"""
        return self.distribution.entropy().sum(dim=-1) if self.distribution else None

    def update_distribution(self, observations, hist_encoding=False):
        """更新动作分布（兼容rsl_rl_old接口）"""
        mean = self.student_actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.student_std)

    def act(self, observations, hist_encoding=False, **kwargs):
        """学生策略执行动作（训练时使用）"""
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """获取动作对数概率（兼容rsl_rl_old接口）"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        """学生策略推理（部署时使用）"""
        actions_mean = self.student_actor(observations, hist_encoding, eval, scandots_latent)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """Critic网络评估状态价值"""
        value = self.student_critic(critic_observations)
        return value
    
    def get_teacher_actions(self, observations, terrain_ids=None, hist_encoding=False, env=None, **kwargs):
        """获取教师动作（用于蒸馏训练）- 完全按照play.py的方式
        
        Args:
            observations: 观测
            terrain_ids: 地形ID，用于路由到对应教师 
            hist_encoding: 是否使用历史编码
            env: 环境实例（暂未使用）
            
        Returns:
            教师动作张量（与play.py输出完全一致）
        """
        if not self.loaded_teachers:
            warnings.warn("教师模型未加载，返回零动作")
            return torch.zeros(observations.shape[0], self.num_actions, device=observations.device)
        
        batch_size = observations.shape[0]
        device = observations.device
        
        # 确保教师网络处于评估模式（与play.py一致）
        for teacher_actor in self.teacher_actors:
            teacher_actor.eval()
        
        # 初始化教师动作
        teacher_actions = torch.zeros(batch_size, self.num_actions, device=device)
        
        with torch.no_grad():  # 教师推理不需要梯度
            if terrain_ids is not None:
                # 根据地形ID路由到对应教师
                terrain_ids = terrain_ids.long()
                terrain_ids = torch.clamp(terrain_ids, 0, self.num_teachers - 1)
                
                for teacher_id in range(self.num_teachers):
                    mask = (terrain_ids == teacher_id)
                    if mask.any():
                        obs_subset = observations[mask]
                        
                        # 关键修复：完全按照play.py的方式调用
                        # play.py: policy = ppo_runner.get_inference_policy(device=env.device)
                        # play.py: actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
                        # 其中policy实际上就是actor_critic.act_inference方法
                        
                        # 注意：play.py中没有传递eval参数，默认为False
                        teacher_action = self.teacher_actors[teacher_id].act_inference(
                            obs_subset.detach(),  # 与play.py一致，使用detach()
                            hist_encoding=hist_encoding,  # 传递hist_encoding参数
                            scandots_latent=None  # 没有深度潜在特征
                            # 注意：没有eval参数，使用默认值False
                        )
                        teacher_actions[mask] = teacher_action
            else:
                # 如果没有地形信息，使用第一个教师
                # 完全按照play.py的调用方式
                teacher_actions = self.teacher_actors[0].act_inference(
                    observations.detach(),  # 与play.py一致，使用detach()
                    hist_encoding=hist_encoding,  # 传递hist_encoding参数  
                    scandots_latent=None  # 没有深度潜在特征
                    # 注意：没有eval参数，使用默认值False
                )
        
        # 简单的调试信息
        # print(f"[DEBUG] 教师0原始输出: [{teacher_actions.min().item():.4f}, {teacher_actions.max().item():.4f}]")
        # print(f"[DEBUG] 最终教师动作输出: [{teacher_actions.min().item():.4f}, {teacher_actions.max().item():.4f}]")
        # print(f"[DEBUG] 教师动作均值: {teacher_actions.mean().item():.4f}, 标准差: {teacher_actions.std().item():.4f}")
        
        return teacher_actions
    
    def get_terrain_adaptive_weights(self, terrain_ids, temperature=1.0):
        """计算地形自适应权重（用于加权蒸馏损失）
        
        Args:
            terrain_ids: 地形ID张量
            temperature: 温度参数，控制权重的锐利程度
            
        Returns:
            权重张量 [batch_size, num_teachers]
        """
        batch_size = terrain_ids.shape[0]
        device = terrain_ids.device
        
        # 创建one-hot编码
        weights = torch.zeros(batch_size, self.num_teachers, device=device)
        terrain_ids = terrain_ids.long()
        terrain_ids = torch.clamp(terrain_ids, 0, self.num_teachers - 1)
        
        # 设置对应地形的权重为1
        weights.scatter_(1, terrain_ids.unsqueeze(1), 1.0)
        
        # 应用温度缩放
        if temperature != 1.0:
            weights = torch.softmax(weights / temperature, dim=1)
        
        return weights
    
    def reset_std(self, std, num_actions, device):
        """重置动作标准差（兼容rsl_rl_old接口）"""
        new_std = std * torch.ones(num_actions, device=device)
        self.student_std.data = new_std.data


# 导出类
__all__ = ["MultiTeacherStudent"]