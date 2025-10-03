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
import os           # 操作系统接口
import copy         # 深拷贝功能
import torch        # PyTorch深度学习框架
import numpy as np  # 数值计算库
import random       # 随机数生成
from isaacgym import gymapi   # Isaac Gym API
from isaacgym import gymutil  # Isaac Gym 工具函数
import argparse     # 命令行参数解析
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 项目路径常量

def class_to_dict(obj) -> dict:
    """
    将类对象转换为字典格式
    递归处理嵌套的类属性
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的字典
    """
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):  # 跳过私有属性
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):  # 处理列表类型
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)  # 递归处理嵌套对象
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    """
    从字典更新类对象的属性
    
    Args:
        obj: 要更新的对象
        dict: 包含新值的字典
    """
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):  # 如果属性是类类型，递归更新
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)  # 直接设置属性值
    return

def set_seed(seed):
    """
    设置所有随机数生成器的种子，确保实验可复现
    
    Args:
        seed: 随机种子值，如果为-1则随机生成
    """
    if seed == -1:
        seed = np.random.randint(0, 10000)  # 随机生成种子
    print("Setting seed: {}".format(seed))
    
    # 设置各种随机数生成器的种子
    random.seed(seed)              # Python内置随机数
    np.random.seed(seed)           # NumPy随机数
    torch.manual_seed(seed)        # PyTorch CPU随机数
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希随机化
    torch.cuda.manual_seed(seed)   # PyTorch单GPU随机数
    torch.cuda.manual_seed_all(seed)  # PyTorch多GPU随机数

def parse_sim_params(args, cfg):
    """
    解析仿真参数配置
    基于Isaac Gym Preview 2的代码
    
    Args:
        args: 命令行参数
        cfg: 配置字典
        
    Returns:
        配置好的仿真参数对象
    """
    # 初始化仿真参数
    sim_params = gymapi.SimParams()

    # 根据命令行参数设置值
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu          # 是否使用GPU
        sim_params.physx.num_subscenes = args.subscenes  # 子场景数量
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline  # GPU管道

    # 如果配置中提供了仿真选项，解析并更新/覆盖上述设置
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果命令行传入了线程数，覆盖默认设置
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    """
    获取模型加载路径
    支持模糊匹配运行名称和自动选择最新检查点
    
    Args:
        root: 根目录路径
        load_run: 运行编号，-1表示最新
        checkpoint: 检查点编号，-1表示最新
        model_name_include: 模型文件名包含的字符串
        
    Returns:
        模型文件的完整路径
    """
    # 如果指定了load_run，需要找到对应的目录
    if load_run != -1:
        # 从root的父目录中查找load_run对应的目录
        parent_dir = os.path.dirname(root)
        if os.path.exists(parent_dir):
            # 查找包含load_run名称的目录
            for name in os.listdir(parent_dir):
                if load_run in name and os.path.isdir(os.path.join(parent_dir, name)):
                    root = os.path.join(parent_dir, name)
                    break
    
    # 如果根目录不存在，尝试使用前6个字符匹配运行名称
    if not os.path.isdir(root):
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        if os.path.exists(model_parent):
            model_names = os.listdir(model_parent)
            model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
            for name in model_names:
                if len(name) >= 6:
                    if name[:6] == model_name_cand:  # 前6个字符匹配
                        root = os.path.join(model_parent, name)
                        break
                    
    if checkpoint==-1:  # 自动选择最新检查点
        models = [file for file in os.listdir(root) if model_name_include in file]
        if not models:  # 如果没有找到模型文件
            raise FileNotFoundError(f"在目录 {root} 中没有找到包含 '{model_name_include}' 的模型文件")
        models.sort(key=lambda m: '{0:0>15}'.format(m))  # 按文件名排序
        model = models[-1]  # 选择最新的
    else:  # 指定检查点
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """
    根据命令行参数更新配置
    
    Args:
        env_cfg: 环境配置对象
        cfg_train: 训练配置对象  
        args: 命令行参数对象
        
    Returns:
        更新后的环境配置和训练配置
    """
    # 更新环境配置参数
    if env_cfg is not None:
        # 相机相关配置
        if args.use_camera:
            env_cfg.depth.use_camera = args.use_camera
            
        # 如果使用相机且为无头模式，设置相机特定参数
        if env_cfg.depth.use_camera and args.headless:
            env_cfg.env.num_envs = env_cfg.depth.camera_num_envs              # 相机模式下的环境数量
            env_cfg.terrain.num_rows = env_cfg.depth.camera_terrain_num_rows  # 地形行数
            env_cfg.terrain.num_cols = env_cfg.depth.camera_terrain_num_cols  # 地形列数
            env_cfg.terrain.max_error = env_cfg.terrain.max_error_camera      # 最大误差
            env_cfg.terrain.horizontal_scale = env_cfg.terrain.horizontal_scale_camera  # 水平缩放
            env_cfg.terrain.simplify_grid = True                             # 简化网格
            # 设置地形类型比例
            env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.2    # 跨栏地形比例
            env_cfg.terrain.terrain_dict["parkour_flat"] = 0.05     # 平坦地形比例
            env_cfg.terrain.terrain_dict["parkour_gap"] = 0.2       # 间隙地形比例
            env_cfg.terrain.terrain_dict["parkour_step"] = 0.2      # 台阶地形比例
            env_cfg.terrain.terrain_dict["demo"] = 0.15             # 演示地形比例
            env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
            
        # 相机模式下的Y轴范围限制
        if env_cfg.depth.use_camera:
            env_cfg.terrain.y_range = [-0.1, 0.1]  # 限制Y轴移动范围

        # 基础环境参数覆盖
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs    # 并行环境数量
        if args.seed is not None:
            env_cfg.seed = args.seed                # 随机种子
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows    # 地形网格行数
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols    # 地形网格列数
            
        # 动作延迟配置
        if args.delay:
            env_cfg.domain_rand.action_delay = args.delay
            
        # 从头训练时的默认延迟设置
        if not args.delay and not args.resume and not args.use_camera and args.headless:
            env_cfg.domain_rand.action_delay = True  # 启用动作延迟
            env_cfg.domain_rand.action_curr_step = env_cfg.domain_rand.action_curr_step_scratch
            
    # 更新训练配置参数
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed              # 训练随机种子
            
        # 算法运行器参数
        if args.use_camera:
            cfg_train.depth_encoder.if_depth = args.use_camera  # 深度编码器开关
            
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations  # 最大训练迭代次数
            
        # 恢复训练相关配置
        if args.resume:
            cfg_train.runner.resume = args.resume   # 启用恢复训练
            # 使用恢复训练的特权正则化系数调度
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
            
        # 实验管理参数
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name  # 实验名称
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name              # 运行名称
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run              # 要加载的运行
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint          # 检查点编号

    return env_cfg, cfg_train

def get_args():
    """
    解析命令行参数的函数
    定义了训练和测试过程中可用的所有命令行参数
    """
    custom_parameters = [
        # === 基础训练参数 ===
        {"name": "--task", "type": str, "default": "h1_2_fix", "help": "任务名称，指定要训练的机器人类型和环境"},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "从检查点恢复训练"},
        {"name": "--experiment_name", "type": str,  "help": "实验名称，用于组织和标识不同的实验"},
        {"name": "--run_name", "type": str,  "help": "运行名称，用于标识同一实验下的不同运行"},
        {"name": "--load_run", "type": str,  "help": "当resume=True时要加载的运行名称。如果为-1则加载最新的运行"},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "要加载的模型检查点编号。如果为-1则加载最新的检查点"},
        
        # === 运行环境参数 ===
        {"name": "--headless", "action": "store_true", "default": False, "help": "强制关闭图形显示，适用于服务器训练"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "强化学习算法使用的设备 (cpu, gpu, cuda:0, cuda:1等)"},
        {"name": "--num_envs", "type": int, "help": "并行环境数量，影响训练速度和内存使用"},
        {"name": "--seed", "type": int, "help": "随机种子，用于结果复现"},
        {"name": "--max_iterations", "type": int, "help": "最大训练迭代次数"},
        {"name": "--device", "type": str, "default": "cuda:0", "help": "仿真、强化学习和图形渲染使用的设备"},

        # === 环境布局参数 ===
        {"name": "--rows", "type": int, "help": "环境网格的行数，用于多环境并行训练的布局"},
        {"name": "--cols", "type": int, "help": "环境网格的列数，用于多环境并行训练的布局"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "调试模式，禁用wandb日志记录"},
        {"name": "--proj_name", "type": str,  "default": "parkour_new", "help": "项目名称，用于wandb项目组织和日志文件夹命名"},
        
        # === 实验标识参数 ===
        {"name": "--exptid", "type": str, "help": "实验ID，用于唯一标识当前实验"},
        {"name": "--resumeid", "type": str, "help": "恢复训练时的实验ID"},
        {"name": "--use_camera", "action": "store_true", "default": False, "help": "启用相机渲染，用于知识蒸馏或视觉学习"},

        # === 数据处理参数 ===
        {"name": "--save", "action": "store_true", "default": False, "help": "保存评估数据，用于后续分析"},
        {"name": "--replay", "action": "store_true", "default": False, "help": "重放数据集模式"},
        {"name": "--nodelay", "action": "store_true", "default": False, "help": "移除动作延迟（用于理想化测试）"},
        {"name": "--delay", "action": "store_true", "default": False, "help": "添加动作延迟（模拟真实机器人的响应延迟）"},

        # === 日志记录参数 ===
        {"name": "--no_wandb", "action": "store_true", "default": False, "help": "禁用wandb实验跟踪"}


    ]
    # 解析参数
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # 名称对齐处理
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, name):
    """
    将策略网络导出为JIT格式，用于部署
    
    Args:
        actor_critic: Actor-Critic网络
        path: 导出路径
        name: 文件名
    """
    if hasattr(actor_critic, 'memory_a'):
        # 假设使用LSTM: TODO 添加GRU支持
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name+".pt")
        model = copy.deepcopy(actor_critic.actor).to('cpu')  # 复制到CPU
        traced_script_module = torch.jit.script(model)       # JIT脚本化
        traced_script_module.save(path)                      # 保存模型


class PolicyExporterLSTM(torch.nn.Module):
    """
    LSTM策略导出器
    用于导出包含LSTM记忆的策略网络
    """
    def __init__(self, actor_critic):
        """
        初始化LSTM策略导出器
        
        Args:
            actor_critic: 包含LSTM的Actor-Critic网络
        """
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)          # 复制actor网络
        self.is_recurrent = actor_critic.is_recurrent           # 是否循环网络
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)  # 复制LSTM记忆
        self.memory.cpu()  # 移到CPU
        # 注册隐藏状态和细胞状态缓冲区
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入观测
            
        Returns:
            动作输出
        """
        # LSTM前向传播，更新隐藏状态
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # 更新隐藏状态
        self.cell_state[:] = c    # 更新细胞状态
        return self.actor(out.squeeze(0))  # 通过actor网络输出动作

    @torch.jit.export
    def reset_memory(self):
        """
        重置LSTM记忆状态
        使用@torch.jit.export装饰器使其在JIT模型中可用
        """
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        """
        导出LSTM策略为JIT格式
        
        Args:
            path: 导出路径
        """
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')  # 确保在CPU上
        traced_script_module = torch.jit.script(self)  # JIT脚本化
        traced_script_module.save(path)                # 保存

    
# 覆盖gymutil的设备解析函数
def parse_device_str(device_str):
    """
    解析设备字符串
    
    Args:
        device_str: 设备字符串，如'cpu', 'cuda', 'cuda:0'
        
    Returns:
        设备类型和设备ID的元组
    """
    # 默认值
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id

def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    """
    解析命令行参数
    
    Args:
        description: 程序描述
        headless: 是否支持无头模式
        no_graphics: 是否支持无图形模式
        custom_parameters: 自定义参数列表
        
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description=description)
    
    # 添加基础参数
    if headless:
        parser.add_argument('--headless', action='store_true', help='无头运行，不创建查看器窗口')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='禁用图形上下文创建，不创建查看器窗口，无头部渲染不可用')
    
    # 仿真设备和管道参数
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='物理设备，使用PyTorch语法')
    parser.add_argument('--pipeline', type=str, default="gpu", help='张量API管道 (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='图形设备ID')

    # 物理引擎选择（互斥组）
    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='使用FleX物理引擎')
    physics_group.add_argument('--physx', action='store_true', help='使用PhysX物理引擎')

    # PhysX特定参数
    parser.add_argument('--num_threads', type=int, default=0, help='PhysX使用的核心数')
    parser.add_argument('--subscenes', type=int, default=0, help='并行仿真的PhysX子场景数')
    parser.add_argument('--slices', type=int, help='处理环境切片的客户端线程数')

    # 添加自定义参数
    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:  # 类型参数
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:  # 动作参数
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("错误: 命令行参数必须定义name和type/action，参数未添加到解析器")
            print("支持的键: name, type, default, action, help")
            print()

    args = parser.parse_args()

    # 设备参数处理
    if args.device is not None:
        args.sim_device = args.device
        args.rl_device = args.device
    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    # 管道验证
    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"无效的管道 '{args.pipeline}'。应该是cpu或gpu。"
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    # Flex和设备兼容性检查
    if args.sim_device_type != 'cuda' and args.flex:
        print("无法在CPU上使用Flex。将仿真设备更改为'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    # GPU管道和物理设备兼容性检查
    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("无法在CPU物理上使用GPU管道。将管道更改为'CPU'。")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # 默认使用PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # 使用--nographics意味着--headless
    if no_graphics and args.nographics:
        args.headless = True

    # 切片数默认等于子场景数
    if args.slices is None:
        args.slices = args.subscenes

    return args