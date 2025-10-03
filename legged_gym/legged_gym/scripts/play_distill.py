# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
蒸馏模型测试脚本
专门用于测试多教师蒸馏训练后的学生模型
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from terrain_base.config import terrain_config

import torch
import faulthandler

def play_distill(args):
    """
    蒸馏模型测试函数
    加载蒸馏训练后的学生模型并在环境中运行机器人
    
    Args:
        args: 命令行参数对象
    """
    faulthandler.enable()
    
    # 获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 设置测试环境参数
    env_cfg.env.num_envs = 10  # 并行环境数量
    env_cfg.env.episode_length_s = 1000  # 每个回合的最大时长（秒）
    env_cfg.commands.resampling_time = 60  # 命令重采样时间间隔
    env_cfg.rewards.is_play = True  # 标记为游戏/测试模式
    
    # 设置地形参数 - 测试多种地形
    env_cfg.terrain.num_rows = 5  # 地形网格行数
    env_cfg.terrain.num_cols = 10  # 地形网格列数
    env_cfg.terrain.max_init_terrain_level = 2  # 最大初始地形难度等级
    
    # 设置噪声和域随机化参数
    env_cfg.noise.add_noise = True  # 添加噪声
    env_cfg.domain_rand.randomize_friction = True  # 随机化摩擦系数
    env_cfg.domain_rand.push_robots = False  # 不推机器人
    
    print("🚀 创建测试环境...")
    # 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()  # 获取初始观测（字典格式）
    
    print("📦 加载蒸馏模型...")
    # 加载蒸馏模型
    train_cfg.runner.resume = True
    distill_runner, train_cfg, log_pth = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args, 
        train_cfg=train_cfg, 
        return_log_dir=True
    )
    
    # 获取学生策略（用于推理）
    student_policy = distill_runner.get_inference_policy(device=env.device)
    
    print("🎮 开始测试...")
    print(f"📊 测试环境数量: {env.num_envs}")
    print(f"🗺️  地形类型: {torch.unique(env.env_class)}")
    
    # 初始化动作张量
    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    
    # 统计信息
    total_steps = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    
    # 主循环：运行机器人
    for i in range(10 * int(env.max_episode_length)):
        # 检查观测格式
        if isinstance(obs, dict):
            # 使用学生观测（policy组）
            student_obs = obs["policy"]
        else:
            # 如果是张量格式，直接使用
            student_obs = obs
        
        # 使用学生策略生成动作
        with torch.no_grad():
            actions = student_policy(student_obs.detach())
        
        # 执行动作，获取新的观测和奖励
        obs, _, rewards, dones, infos = env.step(actions.detach())
        
        # 统计信息
        total_steps += 1
        episode_rewards += rewards
        
        # 每1000步打印一次统计信息
        if total_steps % 1000 == 0:
            avg_reward = episode_rewards.mean().item() / 1000
            print(f"步数: {total_steps:6d} | 平均奖励: {avg_reward:6.3f} | 地形类型: {torch.unique(env.env_class).cpu().numpy()}")
            episode_rewards.zero_()
    
    print("✅ 测试完成！")

if __name__ == '__main__':
    args = get_args()
    play_distill(args) 