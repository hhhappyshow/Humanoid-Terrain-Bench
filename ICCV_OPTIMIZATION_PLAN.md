# ICCV 2025 人形机器人地形挑战赛 - 优化方案

##   比赛目标分析

### 评分公式
```
Episode Score = Completion Rate × 0.9 + (1 - Efficiency Score) × 0.1
```

**优化优先级**：
1. **首要目标**: 最大化完成率（90%权重）→ 走得更远
2. **次要目标**: 提高时间效率（10%权重）→ 走得更快

### 测试场景
- **鲁棒性**: 长距离地形（20+台阶）
- **极限**: 超高难度地形
- **泛化**: 混合地形组合


###   策略1：基于高度信息的速度生成

#### 1.1 当前问题分析
```python
# 当前的随机速度采样（第1168行）
self.commands[env_ids, 0] = torch_rand_float(0.0, 1.5, (len(env_ids), 1), device=self.device)
# 问题：完全随机，不考虑地形难度
# 结果：在困难地形上速度过快导致摔倒，在简单地形上速度过慢浪费时间
```

#### 1.2 智能速度生成策略

**核心思想**：根据前方地形的复杂度动态调整速度命令

##### 1.2.1 地形复杂度评估
```python
def _analyze_terrain_complexity(self):
    """分析前方地形复杂度"""
    # 提取前方高度采样点（机器人前方0-1.2米区域）
    forward_heights = self.measured_heights[:, :front_points_num]  # 前方采样点
    
    # 计算地形复杂度指标
    height_variance = torch.var(forward_heights, dim=1)      # 高度方差（起伏程度）
    height_gradient = torch.max(forward_heights, dim=1)[0] - torch.min(forward_heights, dim=1)[0]  # 高度差
    height_roughness = torch.mean(torch.abs(torch.diff(forward_heights, dim=1)), dim=1)  # 粗糙度
    
    # 综合复杂度评分 [0, 1]
    complexity = torch.clamp(
        0.4 * height_variance + 0.4 * height_gradient + 0.2 * height_roughness,
        0.0, 1.0
    )
    return complexity
```

##### 1.2.2 自适应速度生成
```python
def _generate_adaptive_speed(self, env_ids):
    """基于地形复杂度生成自适应速度"""
    complexity = self._analyze_terrain_complexity()[env_ids]
    
    # 速度策略：
    # - 简单地形（complexity < 0.3）：高速前进 [1.0, 1.5] m/s
    # - 中等地形（0.3 ≤ complexity < 0.7）：中速前进 [0.5, 1.0] m/s  
    # - 困难地形（complexity ≥ 0.7）：低速前进 [0.2, 0.5] m/s
    
    base_speed = 1.5 - complexity  # 基础速度：1.5 → 0.5
    speed_range = 0.3 * (1 - complexity)  # 速度范围：简单地形变化大，困难地形变化小
    
    # 在基础速度±范围内随机采样
    min_speed = torch.clamp(base_speed - speed_range, 0.1, 1.4)
    max_speed = torch.clamp(base_speed + speed_range, 0.2, 1.5)
    
    adaptive_speeds = torch_rand_float(
        min_speed.unsqueeze(1), 
        max_speed.unsqueeze(1), 
        (len(env_ids), 1), 
        device=self.device
    ).squeeze(1)
    
    return adaptive_speeds
```

##### 1.2.3 地形类型特殊处理
```python
def _apply_terrain_specific_speed(self, env_ids, adaptive_speeds):
    """针对特定地形类型的速度调整"""
    
    # 台阶地形：更保守的速度
    stair_mask = (self.env_class[env_ids] == 台阶类型ID)
    adaptive_speeds[stair_mask] *= 0.7  # 台阶上减速30%
    
    # 斜坡地形：根据坡度调整
    slope_mask = (self.env_class[env_ids] == 斜坡类型ID)
    # 上坡减速，下坡可以稍快（通过高度梯度判断）
    forward_gradient = self._get_forward_height_gradient()[env_ids]
    slope_speed_factor = torch.clamp(1.0 - 0.5 * forward_gradient, 0.5, 1.2)
    adaptive_speeds[slope_mask] *= slope_speed_factor[slope_mask]
    
    # 障碍物地形：显著减速
    obstacle_mask = (self.env_class[env_ids] == 障碍物类型ID)
    adaptive_speeds[obstacle_mask] *= 0.5  # 障碍物减速50%
    
    return adaptive_speeds
```

##### 1.2.4 完整的智能速度生成
```python
def _resample_commands_intelligent(self, env_ids):
    """智能的命令重采样（替换原有的随机采样）"""
    
    # 🧠 基于高度信息生成自适应速度
    if self.cfg.commands.height_adaptive_speed:  # 新增配置开关
        adaptive_speeds = self._generate_adaptive_speed(env_ids)
        adaptive_speeds = self._apply_terrain_specific_speed(env_ids, adaptive_speeds)
        self.commands[env_ids, 0] = adaptive_speeds
    else:
        # 保留原有的随机采样作为备选
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], 
            self.command_ranges["lin_vel_x"][1], 
            (len(env_ids), 1), device=self.device
        ).squeeze(1) 
```


###   策略2：基于高度的重心平衡

#### 2.1 地形预测性重心调整
```python
def _reward_terrain_anticipatory_balance(self):
    """地形预测性平衡奖励"""
    # 分析即将踩踏的地面高度
    next_step_heights = self._predict_next_footstep_heights()
    
    # 如果下一步是台阶，提前调整重心
    step_height_diff = next_step_heights[:, 1] - next_step_heights[:, 0]  # 左右脚高度差
    
    # 上台阶时鼓励重心前移
    upward_step = step_height_diff > 0.05
    target_com_x = torch.where(upward_step, 0.05, 0.0)  # 重心前移5cm
    
    current_com_x = self._estimate_center_of_mass()[:, 0]
    com_error = torch.abs(current_com_x - target_com_x)
    return torch.exp(-com_error / 0.02)
```


###   策略3：基于高度梯度

#### 3.1 高度梯度感知的步长优化
```python
def _reward_gradient_aware_stride(self):
    """高度梯度感知的步长优化"""
    forward_gradient = self._get_forward_height_gradient()
    
    # 根据坡度调整最优步长
    # 平地：大步长，上坡：小步长，下坡：中等步长
    target_stride_length = torch.clamp(0.6 - 0.3 * torch.abs(forward_gradient), 0.3, 0.8)
    
    # 计算当前步长（通过脚部位置估计）
    current_stride = self._estimate_current_stride_length()
    stride_error = torch.abs(current_stride - target_stride_length)
    
    return torch.exp(-stride_error / 0.1)
```



###   策略4：基于高度的学习加速

#### 4.1 地形难度课程学习
```python
def _update_terrain_curriculum_intelligent(self, env_ids):
    """基于高度信息的智能课程学习"""
    # 不仅基于移动距离，还要考虑地形复杂度
    dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
    avg_terrain_complexity = self._get_traversed_terrain_complexity()[env_ids]
    
    # 综合评估：距离 × 复杂度加权
    performance_score = dis_to_origin * (1 + avg_terrain_complexity)
    threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s * 1.2
    
    # 基于综合评估调整难度
    move_up = performance_score > threshold * 0.8
    move_down = performance_score < threshold * 0.4
    
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
```


###   策略5：提高完成率（稳定性优化）

#### 5.1 增强稳定性奖励
```python
# 文件：legged_gym/envs/h1/h1_2_fix.py
class rewards:
    class scales:
        #   大幅提高稳定性权重
        orientation = -2.0          # 原:-1.0 → 新:-2.0 (防摔倒)
        base_height = -1.0          # 原:-0.5 → 新:-1.0 (保持高度)
        ang_vel_xy = -0.1           # 原:-0.05 → 新:-0.1 (防侧翻)
        
        #   减少激进行为惩罚
        torques = -0.0001           # 原:-0.0002 → 新:-0.0001 (允许更大力矩)
        dof_vel = -0.0005           # 原:-0.001 → 新:-0.0005 (允许更快运动)
        
        #   增加保守行为奖励
        feet_contact_forces = -0.01  # 新增：惩罚过大接触力
        dof_pos_limits = -10.0      # 新增：强烈惩罚关节超限
```

#### 5.2 优化终止条件
```python
# 文件：legged_gym/envs/base/humanoid_robot.py 第428-448行
def check_termination(self):
    #   放宽终止条件，提高容错性
    roll_cutoff = torch.abs(self.roll) > 2.0    # 原:1.5 → 新:2.0
    pitch_cutoff = torch.abs(self.pitch) > 2.0  # 原:1.5 → 新:2.0
    height_cutoff = self.root_states[:, 2] < 0.3  # 原:0.5 → 新:0.3
```

#### 5.3 增加鲁棒性训练
```python
# 文件：legged_gym/envs/base/legged_robot_config.py
class domain_rand:
    #   增加域随机化强度
    randomize_friction = True
    friction_range = [0.1, 2.0]        # 原:[0.5,1.25] → 新:[0.1,2.0]
    
    randomize_base_mass = True  
    added_mass_range = [-3., 3.]       # 原:[-1,1] → 新:[-3,3]
    
    push_robots = True
    max_push_vel_xy = 0.8              # 原:0.5 → 新:0.8
    push_interval_s = 8                # 原:15 → 新:8 (更频繁推动)
```

###   策略6：提高时间效率（性能优化）

#### 6.1 优化速度跟踪
```python
# 文件：legged_gym/envs/h1/h1_2_fix.py
class rewards:
    class scales:
        #   提高速度跟踪权重
        tracking_lin_vel = 2.0      # 原:1.5 → 新:2.0
        tracking_ang_vel = 0.8      # 原:0.5 → 新:0.8
        
        #   鼓励更快的步态
        feet_air_time = 0.2         # 原:0.1 → 新:0.2 (鼓励更大步长)
```


## 📊 监控指标

### 关键KPI
```python
# 在训练过程中重点关注：
1. completion_rate: 目标完成率 (>0.8为优秀)
2. success_rate: 成功率 (>0.9为优秀)  
3. episode_length: episode长度 (越长越好)
4. terrain_level: 地形难度级别 (越高越好)
```

### wandb监控
```python
# 重点监控的指标
wandb.log({
    'Episode_rew/completion_rate': completion_rate,
    'Episode_rew/success_rate': success_rate,
    'Episode_rew/terrain_level': terrain_level,
    'Train/mean_reward': mean_reward,
    'Train/mean_episode_length': mean_episode_length
})
