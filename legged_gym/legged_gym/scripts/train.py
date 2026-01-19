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

import os
import sys
import ctypes

# 设置 LD_LIBRARY_PATH 以解决 libpython3.8.so.1.0 找不到的问题
# 自动检测 conda 环境路径
conda_env = os.environ.get('CONDA_PREFIX')
if not conda_env:
    # 如果 CONDA_PREFIX 不存在，尝试从 sys.executable 推断
    python_path = sys.executable
    if 'anaconda3' in python_path or 'miniconda3' in python_path:
        # 从 /path/to/anaconda3/envs/env_name/bin/python 提取环境路径
        parts = python_path.split('/')
        for i, part in enumerate(parts):
            if part in ['anaconda3', 'miniconda3']:
                if i + 2 < len(parts) and parts[i+1] == 'envs':
                    conda_env = '/'.join(parts[:i+3])
                    break

if conda_env:
    lib_path = os.path.join(conda_env, 'lib')
    libpython_path = os.path.join(lib_path, 'libpython3.8.so.1.0')
    if os.path.exists(libpython_path):
        # 设置环境变量（对子进程有效）
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path
        
        # 预加载库（对当前进程有效）
        try:
            if sys.platform.startswith('linux'):
                # 使用 RTLD_GLOBAL 确保符号对后续加载的库可见
                ctypes.CDLL(libpython_path, mode=ctypes.RTLD_GLOBAL)
        except (OSError, AttributeError):
            # 如果预加载失败（可能库已加载或权限问题），至少环境变量已设置
            # 这对于通过 dlopen 加载的库仍然有效
            pass

import numpy as np
from datetime import datetime

print("Importing legged_gym modules...")
from legged_gym.envs import *
print("legged_gym.envs imported")
from legged_gym.utils import get_args, task_registry
print("legged_gym.utils imported")
import wandb
print("All imports completed")

def train(args):
    # 不再强制设置 headless，允许通过命令行参数控制
    # 默认情况下不使用 --headless 标志时，args.headless 为 False（显示窗口）
    # 使用 --headless 标志时，args.headless 为 True（无窗口模式）
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid
    os.environ["WANDB_BASE_URL"]='https://api.bandw.top'
    try:
        os.makedirs(log_pth)
    except:
        pass
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
    
    # 初始化 wandb（添加调试信息和错误处理）
    print("Initializing wandb...")
    try:
        wandb.init(project=args.proj_name, name=args.exptid, entity="hhhappyshow-institution", group=args.exptid[:3], mode=mode, dir="../../logs")
        print("wandb.init() completed")
        if mode != "disabled":
            print("Saving files to wandb...")
            wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
            wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
            print("wandb.save() completed")
    except Exception as e:
        print(f"Warning: wandb initialization/save failed ({e}), continuing without wandb...")
        try:
            wandb.init(mode="disabled")
        except:
            pass

    print("Creating environment...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("Environment created successfully")
    
    print("Creating PPO runner...")
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    print("PPO runner created, starting training...")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
