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
import numpy as np  # 数值计算库
import os  # 操作系统接口
from datetime import datetime  # 日期时间处理

# 导入项目相关模块
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 导入路径常量
from legged_gym.envs import *  # 导入所有环境类
from legged_gym.utils import get_args, task_registry  # 导入参数解析和任务注册工具
import wandb  # 权重和偏置(Weights & Biases)实验跟踪工具

def train(args):
    """
    训练函数 - 执行强化学习训练过程
    
    Args:
        args: 命令行参数对象，包含训练配置信息
    """
    # 设置为无头模式(不显示图形界面)
    args.headless = True
    
    # 创建日志路径 - 格式：logs/项目名/月日_时-分-秒--实验ID
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid
    
    # 设置Weights & Biases的API地址 - 注释掉自定义地址，使用标准wandb服务
    # os.environ["WANDB_BASE_URL"]='https://api.bandw.top'
    
    # 尝试创建日志目录
    try:
        os.makedirs(log_pth)
    except:
        pass  # 如果目录已存在则忽略错误
    
    # 根据是否为调试模式设置wandb模式
    if args.debug:
        mode = "disabled"  # 调试模式下禁用wandb
        # 调试模式下使用较小的环境配置
        args.rows = 10      # 环境网格行数
        args.cols = 8       # 环境网格列数
        args.num_envs = 64  # 并行环境数量
    else:
        mode = "online"     # 正常模式下启用在线同步
    
    # 如果指定不使用wandb，则禁用
    if args.no_wandb:
        mode = "disabled"
    
    # 初始化Weights & Biases实验跟踪
    # project: 项目名称
    # name: 实验名称
    # entity: 用户/团队名称 - 使用默认entity（当前登录用户）
    # group: 实验分组(取实验ID前3个字符)
    # mode: 运行模式(online/disabled)
    # dir: 日志保存目录
    wandb.init(project=args.proj_name, name=args.exptid, group=args.exptid[:3], mode=mode, dir="../../logs")
    
    # 保存重要的配置文件到wandb，便于实验复现
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")  # 保存机器人配置文件
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")        # 保存机器人基类文件

    # 根据任务名称创建环境和环境配置
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # env: HumanoidRobot类的实例 (来自 legged_gym.envs.base.humanoid_robot)
    # env_cfg: H1_2FixCfg类的实例 (来自 legged_gym.envs.h1.h1_2_fix)
    
    # 创建PPO算法运行器和训练配置
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    # ppo_runner: OnPolicyRunner类的实例 (来自 rsl_rl.runners.on_policy_runner)
    # train_cfg: H1_2FixCfgPPO类的实例 (来自 legged_gym.envs.h1.h1_2_fix)
    
    # 开始训练过程
    # num_learning_iterations: 最大训练迭代次数
    # init_at_random_ep_len: 是否在随机episode长度处初始化
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

# 程序入口点
if __name__ == '__main__':
    # 解析命令行参数
    args = get_args()
    # 开始训练
    train(args)
