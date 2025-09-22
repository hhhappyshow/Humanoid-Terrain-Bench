def compute_observations(self):
    """构建观测向量"""
    
    # === 基础本体感知 (51维) ===
    obs_buf = torch.cat([
        self.base_ang_vel * self.obs_scales.ang_vel,      # 3维: 角速度
        imu_obs,                                          # 2维: roll, pitch
        self.delta_yaw[:, None],                          # 1维: 目标偏航角误差
        self.delta_next_yaw[:, None],                     # 1维: 下一目标偏航角误差
        
        # === 关键: 速度指令输入到观测中 ===
        self.commands[:, 0:1],                            # 1维: x方向速度指令 ⭐
    
        (self.env_class != 17).float()[:, None],          # 1维: 环境类别
        (self.env_class == 17).float()[:, None],          # 1维: 环境类别
        
        (self.dof_pos - self.default_dof_pos_all),        # 19维: 关节位置偏差
        self.dof_vel,                                     # 19维: 关节速度
        self.action_history_buf[:, -1],                   # 19维: 上一步动作
        self.contact_filt.float()-0.5,                    # 2维: 足部接触
    ], dim=-1)
       
    # === 地形感知 (132维) ===
    heights = torch.clip(
        self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, 
        -1, 1.
    )
    
    # === 特权信息 (在训练时可见) ===
    priv_explicit = torch.cat([
        self.base_lin_vel * self.obs_scales.lin_vel,     # 9维: 实际速度 (特权)
        0 * self.base_lin_vel,                           # 占位符
        0 * self.base_lin_vel                            # 占位符
    ], dim=-1)
    
    priv_latent = torch.cat([
        self.mass_params_tensor,                         # 4维: 质量参数
        self.friction_coeffs_tensor,                     # 1维: 摩擦系数
        self.motor_strength[0] - 1,                      # 12维: 电机强度1
        self.motor_strength[1] - 1                       # 12维: 电机强度2
    ], dim=-1)
    
    # === 历史信息 (530维) ===
    # 10步历史 × 53维观测 = 530维
    
    # 最终观测向量
    self.obs_buf = torch.cat([
        obs_buf,           # 51维: 本体感知 + 速度指令
        heights,           # 132维: 地形高度
        priv_explicit,     # 9维: 实际速度 (特权)
        priv_latent,       # 29维: 物理参数 (特权)
        history_obs        # 530维: 历史观测
    ], dim=-1)
    
    return self.obs_buf  # 总计: 816维


# #########################################################


# 对于人形机器人(如H1-2)
actions = [
    left_hip_yaw, left_hip_roll, left_hip_pitch,     # 左髋关节 3维
    left_knee, left_ankle_pitch, left_ankle_roll,    # 左腿下肢 3维
    right_hip_yaw, right_hip_roll, right_hip_pitch,  # 右髋关节 3维  
    right_knee, right_ankle_pitch, right_ankle_roll  # 右腿下肢 3维
]

# 动作映射到关节目标角度
target_angle = action_scale * action + default_angle