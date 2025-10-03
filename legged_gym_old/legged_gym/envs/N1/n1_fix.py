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
        # 大幅降低刚度，让动作更柔和自然
        stiffness = {'hip_yaw': 50,      # 从90降到50
                     'hip_roll': 60,     # 从120降到60
                     'hip_pitch': 80,    # 从180降到80
                     'knee': 60,         # 从120降到60
                     'ankle': 25,        # 从45降到25
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 4,       # 从8降到4
                     'hip_roll': 5,      # 从10降到5
                     'hip_pitch': 6,     # 从10降到6
                     'knee': 4,          # 从8降到4
                     'ankle': 1.5,       # 从2.5降到1.5
                     }  # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2               # 从0.25降到0.2，限制动作幅度
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
            # 基础运动奖励
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.8
            
            # 姿态控制 - 进一步降低
            base_height = -0.5          # 从-1.0降到-0.5
            orientation = -0.1          # 从-0.2降到-0.1
            lin_vel_z = -0.5            # 从-1.0降到-0.5
            ang_vel_xy = -0.01          # 从-0.02降到-0.01
            
            # 动作平滑性
            action_rate = -0.1          # 增强平滑性约束
            stand_still = -0.01
            dof_vel = -1e-5
            dof_acc = -5e-9
            dof_pos_limits = -1.0
            dof_vel_limits = -1e-4
            dof_power = -5e-6
            
            # 步态相关 - 最小化
            feet_ground_parallel = -0.005
            feet_clearance = -0.01      # 几乎不惩罚抬脚
            feet_air_time = 0.0         # 完全禁用
            collision = -0.2
            
            # 导航奖励
            _reward_reach_goal = 1.0
            heading_tracking = 0.3
            next_heading_tracking = 0.3
            
            # 步态约束奖励
            foot_contact_sequence = 3.0  # 强制至少一脚着地
            gait_frequency = 0.5         # 轻微鼓励合理步频


        # 参数设置
        target_feet_height = 0.02       # 最小抬脚高度
        feet_air_time_target = 0.02     # 几乎不鼓励腾空    
        base_height_target = 0.7
        max_contact_force = 300.        # 允许更大接触力
        is_play = False
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.


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

