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

# 导入必要的库和模块
from legged_gym import LEGGED_GYM_ROOT_DIR  # 项目根目录路径
import os  # 操作系统接口

from legged_gym.envs import *  # 导入所有环境类
from legged_gym.utils import  get_args,  task_registry  # 导入工具函数和任务注册表
from terrain_base.config import terrain_config  # 导入地形配置

import torch  # PyTorch深度学习框架
import faulthandler  # 故障处理器，用于调试

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    """
    获取模型加载路径的函数
    
    Args:
        root: 模型文件所在的根目录
        load_run: 要加载的运行编号，-1表示加载最新的
        checkpoint: 要加载的检查点编号，-1表示加载最新的
        model_name_include: 模型文件名包含的关键词
        
    Returns:
        model: 模型文件名
        checkpoint: 检查点编号
    """
    if checkpoint==-1:
        # 如果未指定检查点，自动找到最新的模型文件
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))  # 按文件名排序
        model = models[-1]  # 选择最新的模型
        checkpoint = model.split("_")[-1].split(".")[0]  # 从文件名提取检查点编号
    return model, checkpoint

def play(args):
    """
    主要的游戏/测试函数
    加载训练好的模型并在环境中运行机器人
    
    Args:
        args: 命令行参数对象
    """
    faulthandler.enable()  # 启用故障处理器，便于调试
    
    # 获取实验ID和日志路径
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    # 获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 为测试覆盖一些参数
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0  # 禁用动作延迟

    # 设置测试环境参数
    env_cfg.env.num_envs = 10  # 并行环境数量
    env_cfg.env.episode_length_s = 1000  # 每个回合的最大时长（秒）
    env_cfg.commands.resampling_time = 60 # 命令重采样时间间隔
    env_cfg.rewards.is_play = True  # 标记为游戏/测试模式

    # 设置地形参数
    env_cfg.terrain.num_rows = 5  # 地形网格行数
    env_cfg.terrain.num_cols = 10  # 地形网格列数
    env_cfg.terrain.max_init_terrain_level = 2  # 最大初始地形难度等级

    # 设置地形高度范围
    env_cfg.terrain.height = [0.01, 0.02]
    
    # 设置深度相机参数
    env_cfg.depth.angle = [0, 1]
    
    # 设置噪声和域随机化参数
    env_cfg.noise.add_noise = True  # 添加噪声
    env_cfg.domain_rand.randomize_friction = True  # 随机化摩擦系数
    env_cfg.domain_rand.push_robots = False  # 不推机器人
    env_cfg.domain_rand.push_interval_s = 8  # 推机器人间隔
    env_cfg.domain_rand.randomize_base_mass = False  # 不随机化基座质量
    env_cfg.domain_rand.randomize_base_com = False  # 不随机化基座质心

    depth_latent_buffer = []  # 深度潜在特征缓冲区
    
    # 准备环境
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()  # 获取初始观测

    # 加载策略模型
    train_cfg.runner.resume = True  # 设置为恢复模式
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root = log_pth, 
        env=env, 
        name=args.task, 
        args=args,
        # runner_class_name = "MultiTeacherDistillationRunner"
        runner_class_name = "OnPolicyRunner"
    )
    ppo_runner.load("/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/slope7/model_15000.pt")
    # ppo_runner.load("/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/legged_gym/scripts/logs/h1_2_fix_1/Sep30_21-47-48/model_0.pt")
    
    # assert False
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=env.device)
    # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    
    # 如果使用深度相机，获取深度编码器
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    # 初始化动作张量
    actions = torch.zeros(env.num_envs, 19, device=env.device, requires_grad=False)
    infos = {}
    # print(f"actions: {actions}")
    # assert False

    # actions = policy(obs.detach(), hist_encoding=True, eval=False)
    # print(f"shape of actions: {actions.shape}")
    
    # 获取深度信息
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    # 主循环：运行机器人
    for i in range(10*int(env.max_episode_length)):
       
        # 如果使用深度相机
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                # 准备学生观测（去除深度信息）
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0  # 清零深度相关观测
                
                # 使用深度编码器处理深度信息
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]  # 深度潜在特征
                yaw = depth_latent_and_yaw[:, -2:]  # 偏航角信息
                
            # 更新观测中的偏航角信息
            obs[:, 6:8] = 1.5*yaw
                
        else:
            depth_latent = None
        
        # 根据是否有深度actor选择不同的策略
        if hasattr(ppo_runner.alg, "depth_actor"):
            print(f"using depth actor")
            # 使用深度actor
            actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            # 使用普通策略
            ###############################################################################
            #############测试相同obs下教师网络输出与原本的教师模型输出是否一致#################
            "1、随机化obs，看教师网络输出与原本的教师模型输出是否一致"
            import numpy as np

            obs_679 = np.array([
                [ 8.0546727e-03, -3.6421635e-03, -7.0213368e-03,  5.2526465e-04,
                -2.0166329e-04,  0.0000000e+00,  4.5037613e-04,  4.6616539e-04,
                0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
                0.0000000e+00,  3.0242365e-03,  2.2277385e-03, -1.2709423e-03,
                -6.6676140e-03, -2.7855635e-03, -4.8902232e-02, -1.2605737e-03,
                -2.5209486e-03, -3.5682821e-04,  8.0593824e-03,  2.0075887e-03,
                -4.8760224e-02,  9.3834549e-03,  5.2104406e-03, -2.1733518e-03,
                -1.4179296e-02, -5.5518586e-02, -4.9949858e-01, -2.8842248e-03,
                -7.0606768e-03,  1.0877823e-03,  2.6411880e-02, -4.9988132e-02,
                -4.9874470e-01,  1.9601990e-02, -1.3435342e-02, -1.0298284e-01,
                -4.9730211e-02,  7.6743625e-03,  2.1113213e-02, -8.6830556e-04,
                2.9343305e-02,  1.3911884e-02,  5.6031618e-02,  8.9739569e-02,
                2.4020143e-02, -5.0000000e-01, -5.0000000e-01,  6.7131007e-01,
                6.7131007e-01,  6.6631007e-01,  6.7631006e-01,  6.8131006e-01,
                6.7631006e-01,  6.7131007e-01,  6.7631006e-01,  6.7631006e-01,
                6.7631006e-01,  6.7131007e-01,  6.6131008e-01,  6.5631008e-01,
                6.6131008e-01,  6.6131008e-01,  6.5631008e-01,  6.5131009e-01,
                6.5631008e-01,  6.5631008e-01,  6.5631008e-01,  6.5131009e-01,
                6.6631007e-01,  6.4131004e-01,  6.4131004e-01,  6.3631004e-01,
                6.4131004e-01,  6.4131004e-01,  6.4131004e-01,  6.4631009e-01,
                6.4131004e-01,  6.3631004e-01,  6.4131004e-01,  6.3631004e-01,
                6.1631006e-01,  6.1131006e-01,  6.2131006e-01,  6.2131006e-01,
                6.2131006e-01,  6.2131006e-01,  6.1631006e-01,  6.2131006e-01,
                6.1631006e-01,  6.2131006e-01,  6.2131006e-01,  5.9131002e-01,
                6.0131007e-01,  6.0131007e-01,  6.0131007e-01,  5.9131002e-01,
                5.9631008e-01,  6.0131007e-01,  6.0631007e-01,  5.9131002e-01,
                5.9631008e-01,  5.9631008e-01,  5.8631003e-01,  5.8631003e-01,
                5.8631003e-01,  5.8631003e-01,  5.8131003e-01,  5.7631004e-01,
                5.8131003e-01,  5.8631003e-01,  5.7631004e-01,  5.8131003e-01,
                5.7631004e-01,  5.5631006e-01,  5.6631005e-01,  5.6631005e-01,
                5.7131004e-01,  5.6631005e-01,  5.6131005e-01,  5.5631006e-01,
                5.6631005e-01,  5.6631005e-01,  5.6131005e-01,  5.6631005e-01,
                5.3631008e-01,  5.5131006e-01,  5.4131007e-01,  5.4131007e-01,
                5.3631008e-01,  5.4131007e-01,  5.4131007e-01,  5.4631007e-01,
                5.4631007e-01,  5.5131006e-01,  5.5131006e-01,  5.2631009e-01,
                5.3631008e-01,  5.3631008e-01,  5.2631009e-01,  5.2631009e-01,
                5.3131008e-01,  5.2131009e-01,  5.3131008e-01,  5.3131008e-01,
                5.3131008e-01,  5.3631008e-01,  5.0631005e-01,  5.0631005e-01,
                5.1131004e-01,  5.0131005e-01,  5.0631005e-01,  5.0631005e-01,
                5.0631005e-01,  5.1131004e-01,  5.0631005e-01,  5.0131005e-01,
                5.0631005e-01,  4.8631006e-01,  4.9131006e-01,  4.9131006e-01,
                4.8631006e-01,  4.8631006e-01,  4.9631006e-01,  4.8631006e-01,
                4.9131006e-01,  4.8131007e-01,  4.8131007e-01,  4.9631006e-01,
                4.7131005e-01,  4.7131005e-01,  4.8131007e-01,  4.8131007e-01,
                4.8131007e-01,  4.8131007e-01,  4.7131005e-01,  4.7631007e-01,
                4.7631007e-01,  4.7131005e-01,  4.7631007e-01, -2.5763062e-05,
                -7.2046272e-03, -4.9383715e-01, -0.0000000e+00, -0.0000000e+00,
                -0.0000000e+00, -0.0000000e+00, -0.0000000e+00, -0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                8.0000001e-01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00, -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,
                1.5942507e-12, -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  9.3002582e-01,
                1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00, -5.0000000e-01, -5.0000000e-01,
                -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,  1.5942507e-12,
                -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00, -5.0000000e-01, -5.0000000e-01, -4.8428811e-10,
                5.9604610e-10, -1.9208597e-10,  1.5942507e-12, -5.1849698e-09,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  9.3002582e-01,  1.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                -5.0000000e-01, -5.0000000e-01, -4.8428811e-10,  5.9604610e-10,
                -1.9208597e-10,  1.5942507e-12, -5.1849698e-09,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                9.3002582e-01,  1.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -5.0000000e-01,
                -5.0000000e-01, -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,
                1.5942507e-12, -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  9.3002582e-01,
                1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00, -5.0000000e-01, -5.0000000e-01,
                -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,  1.5942507e-12,
                -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00, -5.0000000e-01, -5.0000000e-01, -4.8428811e-10,
                5.9604610e-10, -1.9208597e-10,  1.5942507e-12, -5.1849698e-09,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  9.3002582e-01,  1.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -5.0000000e-01,
                -5.0000000e-01, -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,
                1.5942507e-12, -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  9.3002582e-01,
                1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00, -5.0000000e-01, -5.0000000e-01,
                -4.8428811e-10,  5.9604610e-10, -1.9208597e-10,  1.5942507e-12,
                -5.1849698e-09,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  9.3002582e-01,  1.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                0.0000000e+00, -5.0000000e-01, -5.0000000e-01]
            ])

            print(f"obs shape: {obs.shape}")  # 输出: (1, 731)
            obs_731 = np.zeros((1, 731))
            obs_731[0, :679] = obs_679[0]
            
            batch_size = 10  # 与环境数量匹配
            obs = np.tile(obs_731, (batch_size, 1))
            obs = torch.from_numpy(obs).float().to("cuda")

            print(f"obs_batch shape: {obs.shape}")

########################################################################################
########################################################################################
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            print(f"obs: {obs.detach().shape}")
            print(f"actions before process: {actions[0]}")
        # assert False

#######################################################################
######################测试输出的pt文件是否会输出相同动作########################

        # train_cfg.runner.resume = True  # 设置为恢复模式
        # ppo_runner, train_cfg = task_registry.make_alg_runner(
        #     log_root = log_pth, 
        #     env=env, 
        #     name=args.task, 
        #     args=args,
        #     runner_class_name = "MultiTeacherDistillationRunner"
        #     # runner_class_name = "OnPolicyRunner"
        # )
        # # ppo_runner.load("/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/slope7/model_15000.pt")
        # ppo_runner.load("/home/rashare/zhong/Humanoid-Terrain-Bench-kelun/legged_gym/legged_gym/scripts/logs/h1_2_fix_1/Sep30_21-47-48/model_0.pt")

        # policy = ppo_runner.get_inference_policy(device=env.device)
        # actions = policy(obs.detach(), hist_encoding=True, eval=False)
        # print(f"actions: {actions}")
        # assert False


#######################################################################
#######################################################################
        # 添加调试信息：打印动作范围
        if i % 50 == 0:  # 每50步打印一次
            print(f"[PLAY] Step {i}: 原始策略输出 [{actions.min().item():.4f}, {actions.max().item():.4f}]")
            print(f"[PLAY] Step {i}: 原始动作均值 {actions.mean().item():.4f}, 标准差 {actions.std().item():.4f}")
            
        # 执行动作，获取新的观测和奖励
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 打印环境处理后的动作
        if i % 50 == 0:  # 每50步打印一次
            processed_actions = env.actions  # 环境处理后的动作
            print(f"processed_actions: {processed_actions[0]}")
            # assert False

if __name__ == '__main__':
    # 全局配置标志
    EXPORT_POLICY = False  # 是否导出策略
    RECORD_FRAMES = False  # 是否录制帧
    MOVE_CAMERA = False  # 是否移动相机
    
    # 获取命令行参数并运行
    args = get_args()
    play(args)
