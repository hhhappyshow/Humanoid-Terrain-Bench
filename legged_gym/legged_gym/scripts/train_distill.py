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

# 基于rsl_rl_old框架的多教师蒸馏训练脚本

import numpy as np
import os
from datetime import datetime

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils import class_to_dict
import wandb


def main():
    """H1机器人多教师蒸馏训练主函数"""
    
    # ========== 解析命令行参数 ==========
    args = get_args()
    
    # ========== 创建环境并获取观测维度 ==========
    print("创建训练环境...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 获取实际的观测维度 - 现在返回完整的731维观测
    obs = env.get_observations()
    # 现在obs是完整的731维张量，学生和教师都使用相同的观测
    actor_obs = obs
    # 修复：确保critic_obs不为None
    privileged_obs = env.get_privileged_observations() if hasattr(env, 'get_privileged_observations') else None
    critic_obs = privileged_obs if privileged_obs is not None else obs
    
    actual_actor_obs_dim = actor_obs.shape[-1] if hasattr(actor_obs, 'shape') else env.num_obs
    actual_critic_obs_dim = critic_obs.shape[-1] if hasattr(critic_obs, 'shape') else env.num_privileged_obs or env.num_obs
    
    print(f"实际观测维度: Actor={actual_actor_obs_dim}, Critic={actual_critic_obs_dim}")
    
    # ========== 配置训练参数 ==========
    # 教师模型路径列表 - 使用训练好的slope模型进行测试
    teacher_model_paths = [
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/hurdle1/model_50000.pt",      # 教师0: parkour (跑酷0)
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/bridge2/model_164000.pt",     # 教师1: bridge (桥梁2)
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/flat3/model_0.pt",            # 教师2: flat (平地3) - 已训练15000次迭代
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/uneven4/model_73000.pt",      # 教师3: uneven (不平整4)
        # "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/stair5/model_50000.pt",     # 教师4: stair (楼梯5) 
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/wave6/model_0.pt",            # 教师4: wave (波浪6) 
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/slope7/model_15000.pt",       # 教师5: slope (斜坡7)
        "/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/gap8/model_8000.pt",          # 教师6: gap (间隙8) 
    ]
    
    # 验证教师模型路径是否存在
    print("验证教师模型路径...")
    valid_paths = []
    for i, path in enumerate(teacher_model_paths):
        if os.path.exists(path):
            valid_paths.append(path)
            print(f"✓ 教师{i}: {path}")
        else:
            print(f"✗ 教师{i}: {path} (不存在)")
    
    if len(valid_paths) == 0:
        print("警告: 没有找到有效的教师模型路径，将使用随机初始化的教师网络")
        teacher_model_paths = []
        num_teachers = 5
    else:
        print(f"找到 {len(valid_paths)} 个有效的教师模型")
        teacher_model_paths = valid_paths
        num_teachers = len(valid_paths)

    # ========== 创建多教师蒸馏训练配置 ==========
    train_cfg = {
        "runner": {
            "class_name": "MultiTeacherDistillationRunner",  # 关键：让task_registry识别多教师蒸馏模式
            "algorithm_class_name": "MultiTeacherDistillation",
            "policy_class_name": "MultiTeacherStudent", 
            "num_steps_per_env": 24,
            "max_iterations": 10000,
            "save_interval": 200,
            "experiment_name": args.exptid if hasattr(args, 'exptid') else "h1_multi_teacher_distillation",
            "run_name": "multi_terrain_student",
        },
        
        "algorithm": {
            "class_name": "MultiTeacherDistillation",
            "num_learning_epochs": 3,  # 适中的学习轮数
            "num_mini_batches": 4,
            "learning_rate": 1e-4,     # 降低学习率，因为损失值很大
            "max_grad_norm": 1.0,
            # 蒸馏损失配置
            "distillation_loss_coef": 1.0,
            "behavior_cloning_coef": 1.0,  # 主要损失：模仿教师
            "diversity_loss_coef": 0.1,    # 辅助损失：鼓励地形专门化
            # 地形自适应权重
            "terrain_adaptive_weights": True,  # 启用地形自适应
            "weight_temperature": 1.0,
        },
        
        "policy": {
            "class_name": "MultiTeacherStudent",
            # 教师模型配置
            "num_teachers": num_teachers,
            "teacher_model_paths": teacher_model_paths,  # 这个信息会被task_registry用于wandb配置
            # 网络结构配置 - 使用与实际教师模型匹配的结构
            "actor_hidden_dims": [512, 256, 128],  # 匹配教师模型结构
            "critic_hidden_dims": [512, 256, 128], # 匹配教师模型结构
            "teacher_hidden_dims": [512, 256, 128], # 匹配教师模型结构
            "scan_encoder_dims": [128, 64, 32],     # 匹配教师模型结构
            "activation": "elu",
            "init_noise_std": 1.0,
            # 观测标准化
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "teacher_obs_normalization": False,
            # 兼容rsl_rl_old的参数 - 匹配教师模型的priv_encoder结构
            "priv_encoder_dims": [64, 29],  # 匹配教师模型: [64, 20]
            "tanh_encoder_output": False,
        },
        
        # 虚拟的estimator和depth_encoder配置（兼容rsl_rl_old）
        "estimator": {
            "priv_states_dim": 9,  # 3 + 3 + 3 
            "num_prop": env.cfg.env.n_proprio,
            "num_scan": env.cfg.env.n_scan,
            "hidden_dims": [256, 128],
            "learning_rate": 1e-3,
            "train_with_estimated_states": False,
        },
        
        "depth_encoder": {
            "if_depth": False,
            "learning_rate": 1e-3,
            "hidden_dims": [256, 128],
        }
    }

    # ========== 创建日志目录 ==========
    log_root = "logs"
    experiment_name = train_cfg["runner"]["experiment_name"]
    run_name = train_cfg["runner"]["run_name"]
    
    # 添加时间戳
    time_str = datetime.now().strftime('%b%d_%H-%M-%S_')
    run_name = run_name or "default_run"  # 确保run_name不为None
    log_dir = os.path.join(log_root, experiment_name, time_str + run_name)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"日志目录: {log_dir}")

    # ========== Wandb配置（将由task_registry处理） ==========
    # 注意：wandb初始化将由task_registry.make_alg_runner自动处理
    # 这里我们只需要确保args中有正确的项目名称
    print("✓ Wandb配置准备完成，将由task_registry自动初始化")
    print(f"  - 项目名称: {args.proj_name if hasattr(args, 'proj_name') else 'legged_gym'}")
    print(f"  - 实验ID: {args.exptid if hasattr(args, 'exptid') else 'experiment'}")
    print(f"  - 教师模型数量: {num_teachers}")
    print(f"  - 教师模型路径: {teacher_model_paths}")

    # ========== 创建多教师蒸馏运行器 ==========
        
    # 获取注册的配置
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    
    # 修改注册配置中的教师模型路径
    if hasattr(train_cfg, 'policy') and hasattr(train_cfg.policy, 'teacher_model_paths'):
        train_cfg.policy.teacher_model_paths = teacher_model_paths
        train_cfg.policy.num_teachers = num_teachers
        print(f"✓ 已更新注册配置中的教师模型路径: {num_teachers}个教师")
        
    # 使用task_registry创建运行器，这样wandb会被正确初始化
    runner, _ = task_registry.make_alg_runner(
        log_root="logs",
        env=env,
        name=args.task,
        args=args,
        init_wandb=True
    )

    # ========== 开始训练 ==========
    print("开始多教师蒸馏训练...")
    print(f"- 最大迭代次数: {train_cfg.runner.max_iterations}")
    print(f"- 每个环境步数: {train_cfg.runner.num_steps_per_env}")
    print(f"- 并行环境数: {env.num_envs}")
    print(f"- 总步数: {train_cfg.runner.max_iterations * train_cfg.runner.num_steps_per_env * env.num_envs:,}")

    # 执行训练
    try:
        runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True
        )
        print("✓ 训练完成！")
        
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 确保wandb正确结束
        try:
            wandb.finish()
        except:
            pass


if __name__ == '__main__':
    main()
