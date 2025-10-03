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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy

class N1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.70]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 左腿 - 抵消URDF中的默认偏置
            "left_hip_pitch_joint": -numpy.deg2rad(8.0),
            "left_hip_roll_joint": -numpy.deg2rad(15.0),    # 抵消URDF中的+15度偏置
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": +numpy.deg2rad(20.0),
            "left_ankle_roll_joint": 0.0,
            "left_ankle_pitch_joint": -numpy.deg2rad(8.0),

            # 右腿 - 抵消URDF中的默认偏置
            "right_hip_pitch_joint": -numpy.deg2rad(8.0),
            "right_hip_roll_joint": +numpy.deg2rad(15.0),   # 抵消URDF中的-15度偏置
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": +numpy.deg2rad(20.0),
            "right_ankle_roll_joint": 0.0,
            "right_ankle_pitch_joint": -numpy.deg2rad(8.0),
            
            'torso_joint' : 0.0
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 512
        n_scan = 132
        n_priv = 3 + 3 + 3 # = 9 base velocity 3个

        n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
        n_proprio = 51 # 所有本体感知信息，即obs_buf
        history_len = 10

        # num obs = 53+132+10*53+43+9 = 187+47+530+43+9 = 816
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 

        contact_buf_len = 100

        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # 调整刚度参数，平衡稳定性和灵活性
        stiffness = {'hip_yaw': 80,      # 提高髋关节偏航刚度
                     'hip_roll': 100,    # 提高髋关节滚转刚度
                     'hip_pitch': 120,   # 提高髋关节俯仰刚度
                     'knee': 100,        # 提高膝关节刚度
                     'ankle': 40,        # 提高踝关节刚度
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 6,       # 适度提高阻尼
                     'hip_roll': 8,      # 适度提高阻尼
                     'hip_pitch': 10,    # 适度提高阻尼
                     'knee': 8,          # 适度提高阻尼
                     'ankle': 3,         # 适度提高阻尼
                     }  # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.3               # 提高动作幅度，允许更大的关节运动
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N1/N1_rotor.urdf'
        name = "N1"
        foot_name = "foot_roll"
        knee_name = "shank"
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        resampling_time = 1.0         # 命令重采样时间间隔（秒）
        heading_command = True         # 启用朝向命令模式
        ang_vel_clip = 0.1            # 角速度命令死区阈值
        lin_vel_clip = 0.1            # 线速度命令死区阈值
        
        # 策略1：智能速度生成配置
        height_adaptive_speed = True   # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4  # 地形复杂度权重
        speed_gradient_weight = 0.4   # 高度梯度权重  
        speed_roughness_weight = 0.2  # 地形粗糙度权重
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.1, 0.6]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    # class rewards:
    #     class scales:
    #         termination = -0.0
    #         tracking_lin_vel = 2.0
    #         tracking_ang_vel = 0.8
    #         lin_vel_z = -2.0
    #         ang_vel_xy = -0.05
    #         orientation = -0.
    #         torques = -0.00001
    #         dof_vel = -0.
    #         dof_acc = -2.5e-7
    #         base_height = -0. 
    #         feet_air_time =  1.0
    #         collision = -1.
    #         feet_stumble = -0.0 
    #         action_rate = -0.01
    #         stand_still = -0.
            
    #         feet_distance = 2.5

    #     only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     base_height_target = 1.
    #     max_contact_force = 100. # forces above this value are penalized
    #     is_play = False

    class rewards:
        class scales:
            # 基础运动奖励 - 大幅提高
            tracking_lin_vel = 3.0      # 提高线速度跟踪奖励
            tracking_ang_vel = 1.5      # 提高角速度跟踪奖励
            
            # 姿态控制 - 适度惩罚
            base_height = -0.3          # 轻微惩罚高度偏差
            orientation = -0.5          # 适度惩罚姿态偏差
            lin_vel_z = -1.0            # 惩罚垂直速度
            ang_vel_xy = -0.05          # 惩罚滚转俯仰
            
            # 动作平滑性 - 减少过度约束
            action_rate = -0.01         # 减少动作变化惩罚
            stand_still = -0.1          # 惩罚静止不动
            dof_vel = -1e-6             # 轻微惩罚关节速度
            dof_acc = -1e-8             # 轻微惩罚关节加速度
            dof_pos_limits = -10.0      # 强烈惩罚关节限位
            dof_vel_limits = -1.0       # 惩罚速度限位
            torques = -1e-5             # 轻微惩罚力矩
            
            # 步态相关 - 重新启用关键奖励
            feet_air_time = 1.0         # 重新启用腾空时间奖励
            feet_stumble = -2.0         # 惩罚绊倒
            collision = -5.0            # 强烈惩罚碰撞
            
            # 导航奖励 - 修复函数名
            reach_goal = 2.0            # 修复函数名
            heading_tracking = 1.0      # 提高朝向跟踪奖励
            next_heading_tracking = 0.5
            
            # 步态约束奖励
            foot_contact_sequence = 2.0  # 鼓励至少一脚着地
            gait_frequency = 0.3         # 鼓励合理步频


        # 参数设置
        target_feet_height = 0.05       # 提高最小抬脚高度
        feet_air_time_target = 0.3      # 合理的腾空时间目标
        base_height_target = 0.7
        max_contact_force = 300.        
        is_play = False
        only_positive_rewards = True 
        tracking_sigma = 0.25 
        soft_dof_pos_limit = 0.9        # 稍微放宽关节限位
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8


class N1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'n1_fix'
        max_iterations = 100001 # number of policy updates
        save_interval = 500

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = N1FixCfg.env.n_priv
        num_prop = N1FixCfg.env.n_proprio
        num_scan = N1FixCfg.env.n_scan

