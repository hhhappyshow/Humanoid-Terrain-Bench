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

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn

class LeggedRobotCfg(BaseConfig):
    """
    腿式机器人环境配置基类
    定义了所有腿式机器人仿真环境的通用配置参数
    具体的机器人配置类（如H1_2FixCfg）会继承并覆盖这些参数
    """
    
    class play:
        """游戏/测试模式配置"""
        load_student_config = False  # 是否加载学生配置（知识蒸馏相关）
        mask_priv_obs = False       # 是否屏蔽特权观测信息
        
    class env:
        """环境基础配置"""
        num_envs = 6144  # 并行环境数量（默认值，通常会被具体配置覆盖）

        # 观测维度定义
        n_scan = 132              # 激光雷达扫描点数量
        n_priv = 3+3 +3          # 特权信息维度：位置(3) + 速度(3) + 其他(3)
        n_priv_latent = 4 + 1 + 12 +12  # 特权潜在状态维度
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 本体感受信息维度：重力(3) + 命令(2) + 基座状态(3) + 四元数(4) + 关节(36) + 其他(5)
        history_len = 10         # 历史观测长度

        # 总观测维度 = 本体感受 + 扫描 + 历史观测 + 潜在状态 + 特权信息
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_privileged_obs = None  # 特权观测维度（如果不为None，step()会返回critic的特权观测）
        num_actions = 12         # 动作维度（12个关节的目标角度）
        env_spacing = 3.         # 环境间距（米）- 在使用高度场/三角网格时不使用
        send_timeouts = True     # 是否向算法发送超时信息
        episode_length_s = 20    # episode长度（秒）
        obs_type = "og"          # 观测类型

        # 编码相关
        history_encoding = True  # 是否启用历史编码
        reorder_dofs = True     # 是否重新排序关节
        
        # 动作延迟相关（注释掉的配置）
        # action_delay_range = [0, 5]

        # 额外的视觉输入配置
        include_foot_contacts = True  # 是否包含脚部接触信息
        
        # 随机化启动状态
        randomize_start_pos = False    # 是否随机化起始位置
        randomize_start_vel = False    # 是否随机化起始速度
        randomize_start_yaw = False    # 是否随机化起始偏航角
        rand_yaw_range = 1.2          # 偏航角随机范围
        randomize_start_y = False     # 是否随机化Y轴起始位置
        rand_y_range = 0.5            # Y轴随机范围
        randomize_start_pitch = False  # 是否随机化起始俯仰角
        rand_pitch_range = 1.6        # 俯仰角随机范围

        contact_buf_len = 100         # 接触缓冲区长度

        # 目标相关配置
        next_goal_threshold = 0.2     # 下一个目标的阈值距离
        reach_goal_delay = 0.1        # 到达目标的延迟时间
        num_future_goal_obs = 2       # 未来目标观测数量

    class depth:
        """深度相机配置"""
        use_camera = False              # 是否使用深度相机
        camera_num_envs = 192          # 相机模式下的环境数量
        camera_terrain_num_rows = 10   # 相机模式下的地形行数
        camera_terrain_num_cols = 20   # 相机模式下的地形列数

        position = [0.27, 0, 0.03]     # 前置相机位置 [x, y, z]
        angle = [-5, 5]                # 相机角度范围 [min, max] 正值向下俯仰

        update_interval = 5            # 更新间隔（5效果好，8效果较差）

        original = (106, 60)           # 原始图像尺寸 (宽, 高)
        resized = (87, 58)             # 调整后图像尺寸 (宽, 高)
        horizontal_fov = 87            # 水平视野角度
        buffer_len = 2                 # 缓冲区长度
        
        near_clip = 0                  # 近裁剪距离
        far_clip = 2                   # 远裁剪距离
        dis_noise = 0.0                # 距离噪声
        
        scale = 1                      # 深度缩放因子
        invert = True                  # 是否反转深度值

    class normalization:
        """归一化配置"""
        class obs_scales:
            """观测值缩放因子"""
            lin_vel = 2.0              # 线速度缩放
            ang_vel = 0.25             # 角速度缩放
            dof_pos = 1.0              # 关节位置缩放
            dof_vel = 0.05             # 关节速度缩放
            height_measurements = 5.0   # 高度测量缩放
        clip_observations = 100.       # 观测值裁剪范围
        clip_actions = 1.2            # 动作裁剪范围
        
    class noise:
        """噪声配置"""
        add_noise = False             # 是否添加噪声
        noise_level = 1.0            # 噪声级别（缩放其他值）
        quantize_height = True       # 是否量化高度
        class noise_scales:
            """各类噪声的缩放因子"""
            rotation = 0.0           # 旋转噪声
            dof_pos = 0.01          # 关节位置噪声
            dof_vel = 0.05          # 关节速度噪声
            lin_vel = 0.05          # 线速度噪声
            ang_vel = 0.05          # 角速度噪声
            gravity = 0.02          # 重力噪声
            height_measurements = 0.02  # 高度测量噪声

    class commands:
        """运动命令配置"""
        curriculum = True           # 是否启用课程学习
        max_curriculum = 1.         # 最大课程难度
        num_commands = 4            # 命令数量：lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 6.        # 命令重新采样时间间隔（秒）
        heading_command = True      # 是否启用朝向命令（从朝向误差计算角速度命令）
        
        lin_vel_clip = 0.1          # 线速度裁剪
        ang_vel_clip = 0.2          # 角速度裁剪
        
        # 简单模式的命令范围
        class ranges:
            """基础命令范围（课程学习初期）"""
            lin_vel_x = [0., 1.5]      # 前向速度范围 [最小值, 最大值] [m/s]
            lin_vel_y = [0.0, 0.0]     # 侧向速度范围 [m/s]
            ang_vel_yaw = [0, 0]       # 偏航角速度范围 [rad/s]
            heading = [0, 0]           # 朝向角度范围

        # 最大命令范围
        class max_ranges:
            """最大命令范围（课程学习后期或固定模式）"""
            lin_vel_x = [0.3, 0.8]     # 前向速度范围 [m/s]
            lin_vel_y = [-0.3, 0.3]    # 侧向速度范围 [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # 偏航角速度范围 [rad/s]
            heading = [-1.6, 1.6]     # 朝向角度范围

        class crclm_incremnt:
            """课程学习增量"""
            lin_vel_x = 0.1            # 前向速度增量 [m/s]
            lin_vel_y = 0.1            # 侧向速度增量 [m/s]
            ang_vel_yaw = 0.1          # 角速度增量 [rad/s]
            heading = 0.5              # 朝向增量

        waypoint_delta = 0.7           # 路径点间距

    class init_state:
        """初始状态配置"""
        pos = [0.0, 0.0, 1.]          # 初始位置 [x,y,z] [m]
        rot = [0.0, 0.0, 0.0, 1.0]    # 初始旋转四元数 [x,y,z,w]
        lin_vel = [0.0, 0.0, 0.0]     # 初始线速度 [x,y,z] [m/s]
        ang_vel = [0.0, 0.0, 0.0]     # 初始角速度 [x,y,z] [rad/s]
        # 默认关节角度（当动作action = 0.0时的目标角度）
        default_joint_angles = { 
            "joint_a": 0.,             # 关节A的默认角度
            "joint_b": 0.}             # 关节B的默认角度

    class control:
        """控制配置"""
        control_type = 'P'            # 控制类型：P-位置控制，V-速度控制，T-力矩控制
        
        # PD控制器参数：
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # 刚度 [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # 阻尼 [N*m*s/rad]
        
        # 动作缩放：目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25
        # 降采样：每个策略时间步内控制动作更新的次数 @ sim DT
        decimation = 4

    class asset:
        """机器人资产配置"""
        file = ""                     # URDF文件路径
        foot_name = "None"           # 脚部刚体名称，用于索引刚体状态和接触力张量
        penalize_contacts_on = []    # 惩罚接触的部位列表
        terminate_after_contacts_on = []  # 接触后终止的部位列表
        disable_gravity = False      # 是否禁用重力
        collapse_fixed_joints = True # 合并固定关节连接的刚体。特定固定关节可通过添加dont_collapse="true"保留
        fix_base_link = False        # 是否固定机器人基座
        default_dof_drive_mode = 3   # 默认DOF驱动模式（见GymDofDriveModeFlags：0-无，1-位置目标，2-速度目标，3-力矩）
        self_collisions = 0          # 自碰撞设置（1-禁用，0-启用...按位过滤器）
        replace_cylinder_with_capsule = True  # 用胶囊体替换碰撞圆柱体，提高仿真速度/稳定性
        flip_visual_attachments = True        # 某些.obj网格必须从y-up翻转到z-up
        
        # 物理属性
        density = 0.001              # 密度
        angular_damping = 0.         # 角阻尼
        linear_damping = 0.          # 线阻尼
        max_angular_velocity = 1000. # 最大角速度
        max_linear_velocity = 1000.  # 最大线速度
        armature = 0.                # 电枢
        thickness = 0.01             # 厚度

    class domain_rand:
        """域随机化配置 - 提高策略鲁棒性"""
        randomize_friction = True          # 随机化摩擦系数
        friction_range = [0.6, 2.]        # 摩擦系数范围
        randomize_base_mass = True        # 随机化基座质量
        added_mass_range = [0., 3.]       # 额外质量范围 [kg]
        randomize_base_com = True         # 随机化基座质心
        added_com_range = [-0.2, 0.2]     # 质心偏移范围 [m]
        push_robots = True                # 是否施加外部推力
        push_interval_s = 8               # 推力间隔时间 [s]
        max_push_vel_xy = 0.5             # 最大推力速度 [m/s]

        randomize_motor = True            # 随机化电机特性
        motor_strength_range = [0.8, 1.2] # 电机强度范围

        # 动作延迟相关参数
        delay_update_global_steps = 24 * 8000  # 延迟更新的全局步数
        action_delay = False              # 是否启用动作延迟
        action_curr_step = [1, 1]         # 当前动作步数范围
        action_curr_step_scratch = [0, 1] # 从头训练时的动作步数范围
        action_delay_view = 1             # 动作延迟视图
        action_buf_len = 8                # 动作缓冲区长度
        
    class rewards:
        """奖励函数配置"""
        class scales:
            """各项奖励的权重系数"""
            
            # 正则化奖励（通常为负值，起惩罚作用）
            lin_vel_z = -1.0              # 垂直速度惩罚（避免跳跃）
            ang_vel_xy = -0.05            # 滚转俯仰角速度惩罚（保持稳定）
            orientation = -1.             # 姿态惩罚
            dof_acc = -2.5e-7            # 关节加速度惩罚（平滑运动）
            collision = -10.              # 碰撞惩罚
            action_rate = -0.1            # 动作变化率惩罚（平滑控制）
            delta_torques = -1.0e-7       # 力矩变化惩罚
            torques = -0.00001           # 力矩大小惩罚（节能）
            # hip_pos = -0.5             # 髋关节位置惩罚（注释掉）
            dof_error = -0.04            # 关节误差惩罚
            feet_stumble = -1            # 脚部绊倒惩罚
            feet_edge = -1               # 脚部边缘惩罚
            
        only_positive_rewards = True     # 如果为True，负总奖励被裁剪为零（避免早期终止问题）
        tracking_sigma = 0.2            # 跟踪奖励 = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.         # URDF限制的百分比，超过此限制的值会被惩罚
        soft_dof_vel_limit = 1          # 关节速度软限制
        soft_torque_limit = 0.4         # 力矩软限制
        base_height_target = 1.         # 目标基座高度 [m]
        max_contact_force = 40.         # 最大接触力，超过此值会被惩罚 [N]

    # 观察者相机配置
    class viewer:
        """观察者相机设置"""
        ref_env = 0                     # 参考环境ID
        pos = [10, 0, 6]               # 相机位置 [m]
        lookat = [11., 5, 3.]          # 相机朝向点 [m]

    class sim:
        """仿真配置"""
        dt =  0.005                    # 仿真时间步长 [s]
        substeps = 1                   # 子步数
        gravity = [0., 0. ,-9.81]      # 重力加速度 [m/s^2]
        up_axis = 1                    # 上轴方向：0-y轴，1-z轴

        class physx:
            """PhysX物理引擎配置"""
            num_threads = 8                    # 线程数
            solver_type = 1                     # 求解器类型：0-pgs，1-tgs
            num_position_iterations = 4         # 位置迭代次数
            num_velocity_iterations = 0         # 速度迭代次数
            contact_offset = 0.01              # 接触偏移 [m]
            rest_offset = 0.0                  # 静止偏移 [m]
            bounce_threshold_velocity = 0.5    # 弹跳阈值速度 [m/s]
            max_depenetration_velocity = 1.0   # 最大去穿透速度
            max_gpu_contact_pairs = 2**23      # 最大GPU接触对数（2**24用于8000+环境）
            default_buffer_size_multiplier = 5 # 默认缓冲区大小倍数
            contact_collection = 2             # 接触收集：0-从不，1-最后子步，2-所有子步（默认=2）

class LeggedRobotCfgPPO(BaseConfig):
    """
    腿式机器人PPO算法配置基类
    定义了PPO强化学习算法的所有配置参数
    """
    seed = 1                          # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 运行器类名
 
    class policy:
        """策略网络配置"""
        init_noise_std = 1.0              # 初始噪声标准差
        continue_from_last_std = True     # 是否从上次的标准差继续
        scan_encoder_dims = [128, 64, 32] # 扫描编码器维度
        actor_hidden_dims = [512, 256, 128]   # Actor隐藏层维度
        critic_hidden_dims = [512, 256, 128]  # Critic隐藏层维度
        priv_encoder_dims = [64, 20]      # 特权编码器维度
        activation = 'elu'                # 激活函数：elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
        # 仅用于'ActorCriticRecurrent'：
        rnn_type = 'lstm'                 # RNN类型
        rnn_hidden_size = 512             # RNN隐藏层大小
        rnn_num_layers = 1                # RNN层数

        tanh_encoder_output = False       # 编码器输出是否使用tanh激活
    
    class algorithm:
        """PPO算法参数"""
        # 训练参数
        value_loss_coef = 1.0             # 价值损失系数
        use_clipped_value_loss = True     # 是否使用裁剪价值损失
        clip_param = 0.2                  # PPO裁剪参数
        entropy_coef = 0.01               # 熵系数（鼓励探索）
        num_learning_epochs = 5           # 学习轮数
        num_mini_batches = 4              # 小批次数量 = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4             # 学习率
        schedule = 'adaptive'             # 学习率调度：adaptive或fixed
        gamma = 0.99                      # 折扣因子
        lam = 0.95                        # GAE参数
        desired_kl = 0.01                 # 期望KL散度
        max_grad_norm = 1.                # 最大梯度范数
        
        # DAgger参数
        dagger_update_freq = 20           # DAgger更新频率
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]        # 特权正则化系数调度
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]       # 恢复训练时的特权正则化系数调度
    
    class depth_encoder:
        """深度编码器配置"""
        if_depth = LeggedRobotCfg.depth.use_camera    # 是否使用深度
        depth_shape = LeggedRobotCfg.depth.resized    # 深度图像尺寸
        buffer_len = LeggedRobotCfg.depth.buffer_len  # 缓冲区长度
        hidden_dims = 512                             # 隐藏层维度
        learning_rate = 1.e-3                         # 学习率
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24  # 每个环境的步数

    class estimator:
        """状态估计器配置（用于处理特权信息）"""
        train_with_estimated_states = True           # 使用估计状态进行训练
        learning_rate = 1.e-4                        # 学习率
        hidden_dims = [128, 64]                      # 隐藏层维度
        priv_states_dim = LeggedRobotCfg.env.n_priv  # 特权状态维度
        num_prop = LeggedRobotCfg.env.n_proprio      # 本体感受维度
        num_scan = LeggedRobotCfg.env.n_scan         # 扫描数据维度

    class runner:
        """训练运行器配置"""
        policy_class_name = 'ActorCritic'    # 策略类名
        algorithm_class_name = 'PPO'          # 算法类名
        num_steps_per_env = 24               # 每个环境每次迭代的步数
        max_iterations = 100001               # 最大策略更新次数

        # 日志记录
        save_interval = 100                  # 保存间隔（每多少次迭代检查一次保存）
        experiment_name = 'rough_a1'         # 实验名称
        run_name = ''                        # 运行名称
        
        # 加载和恢复
        resume = False                       # 是否恢复训练
        load_run = -1                        # 要加载的运行（-1 = 最后一次运行）
        checkpoint = -1                      # 要加载的检查点（-1 = 最后保存的模型）
        resume_path = None                   # 恢复路径（从load_run和chkpt更新）