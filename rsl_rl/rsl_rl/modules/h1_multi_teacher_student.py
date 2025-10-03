# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
多教师学生策略网络 - H1机器人版本
支持从多个不同地形的教师策略中学习，训练一个通用的学生策略
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import os
import warnings

# 导入RSL-RL的基础模块
try:
    from rsl_rl.modules import ActorCritic
    from rsl_rl.utils import unpad_trajectories
except ImportError:
    print("Warning: RSL-RL modules not found, using placeholder implementations")


class H1MultiTeacherStudent(nn.Module):
    """H1机器人多教师学生策略网络
    
    该网络结构包含：
    1. 学生策略网络：输出实际执行的动作
    2. 多个教师策略网络：基于地形类型提供监督信号
    3. 地形路由机制：根据地形信息选择对应的教师
    """
    
    def __init__(
        self,
        obs_shape: Dict[str, tuple],
        obs_groups: Dict[str, List[str]], 
        num_actions: int,
        num_teachers: int = 6,
        teacher_model_paths: Optional[List[str]] = None,
        student_hidden_dims: List[int] = [512, 256, 128],
        teacher_hidden_dims: List[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        **kwargs
    ):
        """初始化多教师学生网络
        
        Args:
            obs_shape: 观测形状字典
            obs_groups: 观测组配置
            num_actions: 动作维度
            num_teachers: 教师数量
            teacher_model_paths: 教师模型路径列表
            student_hidden_dims: 学生网络隐藏层维度
            teacher_hidden_dims: 教师网络隐藏层维度
            activation: 激活函数名称
            init_noise_std: 初始噪声标准差
            noise_std_type: 噪声类型
            student_obs_normalization: 学生网络是否使用观测标准化
            teacher_obs_normalization: 教师网络是否使用观测标准化
        """
        super().__init__()
        
        # 保存配置
        self.num_teachers = num_teachers
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.teacher_model_paths = teacher_model_paths or []
        
        # 获取观测维度
        self.student_obs_dim = self._get_obs_dim(obs_shape, obs_groups.get("policy", ["policy"]))
        self.teacher_obs_dim = self._get_obs_dim(obs_shape, obs_groups.get("teacher", ["teacher"]))
        
        print(f"学生观测维度: {self.student_obs_dim}")
        print(f"教师观测维度: {self.teacher_obs_dim}")
        
        # 激活函数
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # ==================== 创建学生策略网络 ====================
        student_layers = []
        input_dim = self.student_obs_dim
        
        for hidden_dim in student_hidden_dims:
            student_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation
            ])
            input_dim = hidden_dim
            
        # 输出层：动作均值
        student_layers.append(nn.Linear(input_dim, num_actions))
        self.student_actor = nn.Sequential(*student_layers)
        
        # 动作噪声（学习的标准差）
        if noise_std_type == "scalar":
            self.student_std = nn.Parameter(torch.ones(num_actions) * init_noise_std)
        elif noise_std_type == "diagonal":
            self.student_std = nn.Parameter(torch.ones(num_actions) * init_noise_std)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_std_type}")
        
        # ==================== 创建教师策略网络 ====================
        self.teacher_actors = nn.ModuleList()
        self.teacher_stds = nn.ParameterList()
        
        for i in range(num_teachers):
            # 创建教师网络结构
            teacher_layers = []
            input_dim = self.teacher_obs_dim
            
            for hidden_dim in teacher_hidden_dims:
                teacher_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    self.activation
                ])
                input_dim = hidden_dim
                
            teacher_layers.append(nn.Linear(input_dim, num_actions))
            teacher_actor = nn.Sequential(*teacher_layers)
            
            self.teacher_actors.append(teacher_actor)
            
            # 教师动作噪声
            if noise_std_type == "scalar":
                teacher_std = nn.Parameter(torch.ones(num_actions) * init_noise_std)
            else:
                teacher_std = nn.Parameter(torch.ones(num_actions) * init_noise_std)
            
            self.teacher_stds.append(teacher_std)
        
        # ==================== 观测标准化 ====================
        self.student_obs_normalization = student_obs_normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        
        if student_obs_normalization:
            self.student_obs_rms = RunningMeanStd(shape=(self.student_obs_dim,))
        
        if teacher_obs_normalization:
            self.teacher_obs_rms = RunningMeanStd(shape=(self.teacher_obs_dim,))
        
        # ==================== 加载教师模型 ====================
        self.loaded_teacher = False
        if teacher_model_paths:
            self.load_teacher_models(teacher_model_paths)
        
        # 初始化网络权重
        self._initialize_weights()
        
    def _get_obs_dim(self, obs_shape: Dict[str, tuple], obs_groups: List[str]) -> int:
        """计算观测维度"""
        total_dim = 0
        for group_name in obs_groups:
            if group_name in obs_shape:
                shape = obs_shape[group_name]
                if isinstance(shape, (list, tuple)):
                    total_dim += shape[0] if len(shape) == 1 else shape[-1]
                else:
                    total_dim += shape
        return total_dim
    
    def _initialize_weights(self):
        """初始化网络权重"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        
        self.student_actor.apply(init_weights)
        for teacher_actor in self.teacher_actors:
            teacher_actor.apply(init_weights)
    
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
                # 加载模型状态字典
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 提取actor网络权重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'actor_critic' in checkpoint:
                    state_dict = checkpoint['actor_critic']
                else:
                    state_dict = checkpoint
                
                # 过滤出actor相关的权重
                actor_state_dict = {}
                std_state_dict = {}
                
                for key, value in state_dict.items():
                    if 'actor' in key and 'critic' not in key:
                        # 移除前缀，只保留网络结构
                        new_key = key.replace('actor.', '').replace('actor_critic.actor.', '')
                        if 'std' in new_key:
                            std_state_dict[new_key] = value
                        else:
                            actor_state_dict[new_key] = value
                
                # 加载到对应的教师网络
                missing_keys, unexpected_keys = self.teacher_actors[i].load_state_dict(
                    actor_state_dict, strict=False
                )
                
                if missing_keys:
                    print(f"教师{i}缺少键: {missing_keys}")
                if unexpected_keys:
                    print(f"教师{i}多余键: {unexpected_keys}")
                
                # 加载std参数
                if 'std' in std_state_dict:
                    self.teacher_stds[i].data.copy_(std_state_dict['std'])
                
                print(f"成功加载教师模型 {i}: {model_path}")
                loaded_count += 1
                
            except Exception as e:
                warnings.warn(f"加载教师模型{i}失败: {model_path}, 错误: {str(e)}")
        
        self.loaded_teacher = loaded_count > 0
        print(f"成功加载 {loaded_count}/{self.num_teachers} 个教师模型")
        
        # 冻结教师网络参数
        for teacher_actor in self.teacher_actors:
            for param in teacher_actor.parameters():
                param.requires_grad = False
        
        for teacher_std in self.teacher_stds:
            teacher_std.requires_grad = False
    
    def act(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """学生策略前向传播 - 生成实际执行的动作"""
        student_obs = self._extract_obs(observations, self.obs_groups.get("policy", ["policy"]))
        
        if self.student_obs_normalization:
            student_obs = self.student_obs_rms.normalize(student_obs)
        
        # 学生网络推理
        action_mean = self.student_actor(student_obs)
        
        # 添加噪声
        if self.training:
            noise = torch.randn_like(action_mean) * self.student_std.exp()
            actions = action_mean + noise
        else:
            actions = action_mean
        
        return actions
    
    def evaluate(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """教师策略评估 - 生成教师监督信号"""
        teacher_obs = self._extract_obs(observations, self.obs_groups.get("teacher", ["teacher"]))
        terrain_info = self._extract_obs(observations, self.obs_groups.get("terrain_info", ["terrain_info"]))
        
        if self.teacher_obs_normalization:
            teacher_obs = self.teacher_obs_rms.normalize(teacher_obs)
        
        batch_size = teacher_obs.shape[0]
        device = teacher_obs.device
        
        # 初始化输出动作
        teacher_actions = torch.zeros(batch_size, self.num_actions, device=device)
        
        # 根据地形信息路由到对应教师
        if terrain_info is not None and terrain_info.numel() > 0:
            # 假设terrain_info是地形ID（0-5对应6种地形）
            terrain_ids = terrain_info.squeeze(-1).long()
            terrain_ids = torch.clamp(terrain_ids, 0, self.num_teachers - 1)
            
            for teacher_id in range(self.num_teachers):
                mask = (terrain_ids == teacher_id)
                if mask.any():
                    teacher_obs_subset = teacher_obs[mask]
                    action_mean = self.teacher_actors[teacher_id](teacher_obs_subset)
                    
                    # 教师使用确定性动作（不添加噪声）
                    teacher_actions[mask] = action_mean
        else:
            # 如果没有地形信息，使用第一个教师
            action_mean = self.teacher_actors[0](teacher_obs)
            teacher_actions = action_mean
        
        return teacher_actions
    
    def _extract_obs(self, observations: Dict[str, torch.Tensor], obs_groups: List[str]) -> torch.Tensor:
        """从观测字典中提取指定组的观测"""
        obs_list = []
        for group_name in obs_groups:
            if group_name in observations:
                obs = observations[group_name]
                if len(obs.shape) == 1:
                    obs = obs.unsqueeze(0)
                obs_list.append(obs)
        
        if obs_list:
            return torch.cat(obs_list, dim=-1)
        else:
            # 如果没有找到对应的观测组，返回默认观测
            default_obs = list(observations.values())[0]
            return default_obs
    
    def update_normalization(self, observations: Dict[str, torch.Tensor]):
        """更新观测标准化统计量"""
        if self.student_obs_normalization:
            student_obs = self._extract_obs(observations, self.obs_groups.get("policy", ["policy"]))
            self.student_obs_rms.update(student_obs)
        
        if self.teacher_obs_normalization:
            teacher_obs = self._extract_obs(observations, self.obs_groups.get("teacher", ["teacher"]))  
            self.teacher_obs_rms.update(teacher_obs)
    
    def reset(self, dones: Optional[torch.Tensor] = None):
        """重置网络状态（如果有循环单元）"""
        # H1多教师学生网络是前馈网络，无需重置状态
        pass
    
    def get_hidden_states(self):
        """获取隐藏状态（如果有循环单元）"""
        return None
    
    def detach_hidden_states(self, dones: Optional[torch.Tensor] = None):
        """分离隐藏状态的梯度（如果有循环单元）"""
        pass
    
    def act_inference(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """推理模式下的学生策略前向传播"""
        with torch.no_grad():
            return self.act(observations)


class RunningMeanStd:
    """运行时均值和标准差计算"""
    
    def __init__(self, shape, epsilon=1e-8):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + self.epsilon)


# 为了兼容RSL-RL的命名约定，创建别名
MultiTeacherStudent = H1MultiTeacherStudent