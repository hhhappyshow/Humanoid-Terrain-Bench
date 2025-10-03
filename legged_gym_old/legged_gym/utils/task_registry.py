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

# 导入必要的库
from copy import deepcopy  # 深拷贝功能
import os  # 操作系统接口
from datetime import datetime  # 日期时间处理
from typing import Tuple  # 类型提示
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库

# 导入强化学习相关模块
from rsl_rl.env import VecEnv  # 向量化环境基类
from rsl_rl.runners import OnPolicyRunner  # 在策略强化学习运行器

# 导入项目相关模块
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 项目路径常量
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params  # 辅助函数
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO  # 配置基类
from terrain_base.config import terrain_config  # 地形配置

class TaskRegistry():
    """
    任务注册器类
    用于管理不同机器人任务的注册、配置和创建
    支持环境创建和算法运行器的统一管理
    """
    
    def __init__(self):
        """初始化任务注册器，创建存储字典"""
        self.task_classes = {}  # 存储任务类（环境类）
        self.env_cfgs = {}      # 存储环境配置
        self.train_cfgs = {}    # 存储训练配置
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """
        注册新任务
        
        Args:
            name: 任务名称（如'h1_2_fix'）
            task_class: 环境类
            env_cfg: 环境配置对象
            train_cfg: 训练配置对象
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        """
        获取指定任务的环境类
        
        Args:
            name: 任务名称
            
        Returns:
            对应的环境类
        """
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """
        获取指定任务的配置对象
        
        Args:
            name: 任务名称
            
        Returns:
            环境配置和训练配置的元组
        """
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 同步随机种子
        env_cfg.seed = train_cfg.seed

        # 设置地形配置
        env_cfg.terrain = terrain_config

        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """
        创建环境实例
        
        Args:
            name: 注册的环境名称
            args: Isaac Gym命令行参数，如果为None则调用get_args()
            env_cfg: 环境配置文件，用于覆盖注册的配置
            
        Raises:
            ValueError: 如果没有找到对应名称的注册环境
            
        Returns:
            创建的环境实例和对应的配置文件
        """
        # 如果没有传入参数，获取命令行参数
        if args is None:
            args = get_args()
            
        # 检查是否有对应名称的注册环境
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
            
        if env_cfg is None:
            # 加载配置文件
            env_cfg, _ = self.get_cfgs(name)
        
        # 根据命令行参数覆盖配置（如果指定）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)  # 设置随机种子
        
        # 解析仿真参数（先转换为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        # print("test=",env_cfg.terrain.num_goals)  # 调试信息

        # 创建环境实例
        env = task_class(   cfg=env_cfg,                    # 环境配置
                            sim_params=sim_params,          # 仿真参数
                            physics_engine=args.physics_engine,  # 物理引擎
                            sim_device=args.sim_device,    # 仿真设备
                            headless=args.headless,        # 无头模式
                            save=args.save)                # 保存数据
        # print("test=",env_cfg)  # 调试信息
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, init_wandb=True, log_root="default", **kwargs) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """
        创建训练算法运行器
        
        Args:
            env: 要训练的环境实例
            name: 注册环境的名称，如果为None则使用config文件
            args: Isaac Gym命令行参数，如果为None则调用get_args()
            train_cfg: 训练配置文件，如果为None则使用'name'获取配置
            init_wandb: 是否初始化wandb日志记录
            log_root: Tensorboard日志目录，设为None可避免日志记录（测试时使用）
                     日志保存在<log_root>/<date_time>_<run_name>，默认为<LEGGED_GYM路径>/logs/<experiment_name>
            **kwargs: 其他关键字参数
            
        Raises:
            ValueError: 如果'name'和'train_cfg'都没有提供
            Warning: 如果同时提供'name'和'train_cfg'，'name'将被忽略
            
        Returns:
            创建的PPO算法运行器和对应的配置文件
        """
        # 如果没有传入参数，获取命令行参数
        if args is None:
            args = get_args()
            
        # 如果传入配置文件则使用，否则从名称加载
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 加载配置文件
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
                
        # 根据命令行参数覆盖配置（如果指定）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        
        # 设置日志目录
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = log_root  # os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 将配置转换为字典格式
        train_cfg_dict = class_to_dict(train_cfg)
        
        # 创建在策略运行器
        runner = OnPolicyRunner(env,                    # 环境实例
                                train_cfg_dict,         # 训练配置字典
                                log_dir,                # 日志目录
                                init_wandb=init_wandb,  # 是否初始化wandb
                                device=args.rl_device, # RL设备
                                **kwargs)               # 其他参数
        
        # 在创建新日志目录前保存恢复路径
        resume = train_cfg.runner.resume
        if args.resumeid:
            log_root = LEGGED_GYM_ROOT_DIR + f"/logs/{args.proj_name}/" + args.resumeid
            resume = True
            
        if resume:
            # 加载之前训练的模型
            print(log_root)
            print(train_cfg.runner.load_run)
            # load_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', "rough_a1", train_cfg.runner.load_run)
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            runner.load(resume_path)  # 加载检查点
            
            # 如果不继续使用上次的标准差，则重置
            if not train_cfg.policy.continue_from_last_std:
                runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 12, device=runner.device)

        # 根据参数决定返回内容
        if "return_log_dir" in kwargs:
            return runner, train_cfg, os.path.dirname(resume_path)
        else:    
            return runner, train_cfg

# 创建全局任务注册器实例
task_registry = TaskRegistry()
