# Humanoid Terrain Bench - 工程架构详解

## 🎯 核心训练流程

```
📊 观测 (520维)           🎮 动作 (19维)           🏆 奖励 (标量)
    ↓                        ↓                      ↓
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ 本体感受:72维 │         │ 腿部关节:12维 │         │ 任务奖励:+1.5│
│ 地形高度:396维│   →     │ 手臂关节:7维  │   →     │ 稳定惩罚:-1.0│
│ 特权信息:52维 │         │ (PD控制器)   │         │ 控制惩罚:-0.01│
└─────────────┘         └─────────────┘         └─────────────┘
       ↓                        ↓                      ↓
   Actor-Critic              Isaac Gym              PPO更新
   (策略+价值)                (物理仿真)              (参数优化)
```

## 🏗️ 整体架构

这是一个基于Isaac Gym的人形机器人地形导航强化学习项目，使用PPO算法训练机器人在复杂地形中导航。

### 核心组件
```
Humanoid-Terrain-Bench/
├── 🤖 机器人环境层: HumanoidRobot (legged_gym/envs/base/humanoid_robot.py)
├── 🧠 强化学习算法: PPO (rsl_rl/algorithms/ppo.py)  
├── 🏃 训练管理器: OnPolicyRunner (rsl_rl/runners/on_policy_runner.py)
├── 🗺️ 地形生成: Terrain (challenging_terrain/terrain_base/)
└── ⚙️ 配置文件: LeggedRobotCfg (legged_gym/envs/base/legged_robot_config.py)
```

## 🎯 当前训练模式

**模式：纯教师策略（learn_RL）**
- ✅ 使用高度图地形感知（396个采样点）
- ✅ 使用完整特权信息（真实朝向、线速度等）
- ✅ 4096个并行环境训练
- ❌ 不使用深度相机（仿真研究无需考虑部署）

## 📊 观测空间详解

### 总维度：~520维（具体取决于配置）

#### 1. 本体感受观测 (72维)
```python
# 位置：legged_gym/envs/base/humanoid_robot.py 第672-720行
obs_buf = torch.cat((
    self.base_ang_vel * self.obs_scales.ang_vel,     # 3维: 基座角速度
    imu_obs,                                         # 2维: IMU姿态(roll,pitch)
    0 * self.delta_yaw[:, None],                     # 1维: 占位符
    self.delta_yaw[:, None],                         # 1维: 当前目标朝向误差
    self.delta_next_yaw[:, None],                    # 1维: 下个目标朝向误差
    0 * self.commands[:, 0:2],                       # 2维: 占位符
    self.commands[:, 0:1],                           # 1维: 前向速度命令
    (self.env_class != 17).float()[:, None],        # 1维: 环境类型编码1
    (self.env_class == 17).float()[:, None],         # 1维: 环境类型编码2
    (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,  # 19维: 关节位置偏差
    self.dof_vel * self.obs_scales.dof_vel,          # 19维: 关节速度
    self.action_history_buf[:, -1],                  # 19维: 历史动作
    self.contact_filt.float() - 0.5,                 # 2维: 脚部接触状态
), dim=-1)
```

#### 2. 地形感知观测 (396维)
```python
# 位置：legged_gym/envs/base/humanoid_robot.py 第753行
# 配置：challenging_terrain/terrain_base/config.py 第24-25行
heights = torch.clip(
    self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, 
    -1, 1.
)

# 采样区域：
# X方向: [-0.45, 1.2]米 (机器人后方0.45米到前方1.2米)
# Y方向: [-0.75, 0.75]米 (左右各0.75米)
# 采样点: 12×11×3 = 396个点
```

#### 3. 特权观测 (52维)
```python
# 位置：legged_gym/envs/base/humanoid_robot.py 第730-744行
priv_explicit = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,  # 3维: 真实线速度
    0 * self.base_lin_vel,                        # 3维: 占位符  
    0 * self.base_lin_vel,                        # 3维: 占位符
), dim=-1)  # 总共9维

priv_latent = torch.cat((
    self.mass_params_tensor,      # 4维: 质量和质心参数
    self.friction_coeffs_tensor,  # 1维: 摩擦系数
    self.motor_strength[0] - 1,   # 19维: 电机强度P参数
    self.motor_strength[1] - 1    # 19维: 电机强度D参数
), dim=-1)  # 总共43维
```

#### 4. 历史观测 (72×history_len维)
```python
# 位置：legged_gym/envs/base/humanoid_robot.py 第785-794行
# 滑动窗口存储过去几帧的本体感受观测
self.obs_history_buf = torch.where(
    (self.episode_length_buf <= 1)[:, None, None], 
    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
    torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1)
)
```

## 🎮 动作空间详解

### H1机器人：19维关节控制

```python
# 位置：legged_gym/envs/base/humanoid_robot.py 第958-972行
动作映射 = [
    # 腿部关节 (12维)
    "left_hip_yaw",         # 0:  左髋偏航
    "left_hip_roll",        # 1:  左髋横滚  
    "left_hip_pitch",       # 2:  左髋俯仰
    "left_knee",            # 3:  左膝
    "left_ankle_pitch",     # 4:  左踝俯仰
    "left_ankle_roll",      # 5:  左踝横滚
    "right_hip_yaw",        # 6:  右髋偏航
    "right_hip_roll",       # 7:  右髋横滚
    "right_hip_pitch",      # 8:  右髋俯仰
    "right_knee",           # 9:  右膝
    "right_ankle_pitch",    # 10: 右踝俯仰
    "right_ankle_roll",     # 11: 右踝横滚
    
    # 手臂关节 (7维)
    "left_shoulder_pitch",  # 12: 左肩俯仰
    "left_shoulder_roll",   # 13: 左肩横滚
    "left_shoulder_yaw",    # 14: 左肩偏航
    "left_elbow",           # 15: 左肘
    "right_shoulder_pitch", # 16: 右肩俯仰
    "right_shoulder_roll",  # 17: 右肩横滚
    "right_shoulder_yaw"    # 18: 右肩偏航
]

# 控制方式：PD控制器
# 位置：legged_gym/envs/base/humanoid_robot.py 第958-972行
torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
```

## 🏆 奖励函数详解

### 奖励权重配置
```python
# 配置位置：legged_gym/envs/h1/h1_2_fix.py (H1机器人专用配置)
# 修改位置：继承自 legged_gym/envs/base/legged_robot_config.py

class rewards:
    class scales:
        # 🎯 任务奖励（正值 - 鼓励行为）
        tracking_lin_vel = 1.5      # 线速度跟踪奖励
        tracking_ang_vel = 0.5      # 角速度跟踪奖励  
        feet_air_time = 0.1         # 腾空时间奖励
        
        # ⚠️ 稳定性惩罚（负值 - 惩罚行为）
        orientation = -1.0          # 姿态偏离惩罚
        lin_vel_z = -2.0           # 垂直速度惩罚
        ang_vel_xy = -0.05         # 侧向旋转惩罚
        base_height = -0.5         # 高度偏离惩罚
        
        # 🔧 控制效率惩罚（负值 - 提高效率）
        torques = -0.0002          # 力矩惩罚
        dof_vel = -0.001           # 关节速度惩罚
        action_rate = -0.01        # 动作变化率惩罚
        collision = -1.0           # 碰撞惩罚
        
        # 💀 终止惩罚
        termination = -0.0         # 提前终止惩罚
```

### 奖励函数实现
```python
# 位置：legged_gym/envs/base/humanoid_robot.py 第1989-2080行

def _reward_tracking_lin_vel(self):
    """线速度跟踪奖励 - 鼓励跟踪速度命令"""
    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

def _reward_orientation(self):
    """姿态稳定惩罚 - 惩罚倾倒"""
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

def _reward_feet_air_time(self):
    """步态奖励 - 鼓励自然腾空时间"""
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    # 奖励首次接触地面时的腾空时间
    return torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
```

## ⚙️ 配置修改指南

### 1. 修改观测配置
```python
# 文件：legged_gym/envs/base/legged_robot_config.py

class terrain:
    measure_heights = True          # 是否启用高度图
    measured_points_x = [...]       # X方向采样点
    measured_points_y = [...]       # Y方向采样点
    
class env:
    history_len = 10               # 历史观测长度
    n_proprio = 72                 # 本体感受维度
```

### 2. 修改奖励权重
```python
# 文件：legged_gym/envs/h1/h1_2_fix.py (H1机器人专用)
# 或：legged_gym/envs/base/legged_robot_config.py (通用配置)

class rewards:
    class scales:
        tracking_lin_vel = 1.5     # 调整线速度跟踪重要性
        orientation = -1.0         # 调整姿态稳定重要性
        torques = -0.0002         # 调整能耗惩罚
        # 添加新奖励项...
```

### 3. 修改动作配置
```python
# 文件：legged_gym/envs/base/legged_robot_config.py

class control:
    action_scale = 0.5            # 动作缩放因子
    decimation = 4                # 控制频率降采样
    
    # PD控制器增益
    stiffness = {
        'hip': 80.0,
        'knee': 80.0,
        'ankle': 40.0,
        'shoulder': 40.0,
        'elbow': 40.0
    }
    
    damping = {
        'hip': 2.0,
        'knee': 2.0, 
        'ankle': 1.0,
        'shoulder': 1.0,
        'elbow': 1.0
    }
```

### 4. 修改训练参数
```python
# 文件：legged_gym/envs/base/legged_robot_config.py

class RunnerCfg:
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24        # 每个环境收集步数
    max_iterations = 50000        # 最大训练迭代
    save_interval = 50            # 模型保存间隔
    
class PPOCfg:
    learning_rate = 1.e-3         # 学习率
    num_learning_epochs = 5       # 每次更新的轮数
    num_mini_batches = 4          # mini-batch数量
    clip_param = 0.2              # PPO裁剪参数
```

## 🚀 训练启动

### 基础训练（当前使用）
```bash
# 启动训练
python legged_gym/scripts/train.py --task=h1_2 --headless

# 调试模式（小规模环境）
python legged_gym/scripts/train.py --task=h1_2 --debug

# 继续训练
python legged_gym/scripts/train.py --task=h1_2 --resume --load_run=Sep09_18-20-06--h1-2
```

### 视觉训练（可选）
```bash
# 教师-学生模式训练
python legged_gym/scripts/train.py --task=h1_2 --use_camera --resume --load_run=教师模型路径
```

## 🎮 测试和可视化

### 策略测试
```bash
# 测试训练好的策略
python legged_gym/scripts/play.py --task=h1_2 --load_run=Sep09_18-20-06--h1-2

# 录制回放
python legged_gym/scripts/record_replay.py --task=h1_2 --load_run=Sep09_18-20-06--h1-2
```

### 调试可视化
```python
# 文件：legged_gym/envs/base/humanoid_robot.py 第408-421行
if self.viewer and self.debug_viz:
    self._draw_goals()        # 绘制目标点
    self._draw_height_samples()  # 绘制高度采样点（可选）
```

## 📈 训练监控

### Weights & Biases日志
- 项目名：在 `train.py` 中通过 `--proj_name` 指定
- 监控指标：奖励、损失、成功率、episode长度等

### 本地日志
```
logs/
└── parkour_new/
    └── Sep09_18-20-06--h1-2/
        ├── model_*.pt          # 训练检查点
        └── wandb/              # WandB日志
```

## 🔧 常用修改

### 增加新奖励函数
```python
# 1. 在 humanoid_robot.py 中添加奖励函数
def _reward_your_new_reward(self):
    """您的新奖励描述"""
    # 计算奖励逻辑
    return reward_value

# 2. 在配置文件中添加权重
class rewards:
    class scales:
        your_new_reward = 1.0  # 设置权重
```

### 修改观测维度
```python
# 1. 修改观测拼接逻辑
# 位置：legged_gym/envs/base/humanoid_robot.py compute_observations()

# 2. 更新观测维度配置
# 位置：legged_gym/envs/base/legged_robot_config.py
class env:
    n_proprio = 新的维度数
```

### 调整地形难度
```python
# 文件：challenging_terrain/terrain_base/config.py
terrain_proportions = [
    0.1,  # 平地
    0.15, # 随机粗糙地形
    0.15, # 斜坡
    0.2,  # 台阶
    0.2,  # 离散障碍物
    0.1,  # 波浪地形
    0.1   # 楼梯
]
```

## 📊 性能基准

### 当前训练状态
- **迭代次数**: 17,500 / 50,000
- **并行环境**: 4,096个
- **模型大小**: ~9.24 MB
- **训练模式**: 纯教师策略（高度图）

### 典型性能指标
- **成功率**: 到达所有目标点的比例
- **完成率**: 平均完成目标点的比例  
- **Episode长度**: 平均存活时间
- **奖励**: 各项奖励的加权和

## 🐛 常见问题

### 训练不收敛
1. 检查奖励权重是否合理
2. 降低学习率
3. 检查观测归一化
4. 增加训练环境数量

### 机器人不稳定
1. 增加姿态稳定惩罚权重
2. 检查PD控制器增益
3. 添加力矩惩罚
4. 检查地形难度

### 内存不足
1. 减少并行环境数量
2. 减少历史观测长度
3. 减少地形复杂度

## 📚 代码结构

### 核心文件
- `humanoid_robot.py`: 环境主逻辑，观测计算，奖励函数
- `legged_robot_config.py`: 所有配置参数
- `on_policy_runner.py`: 训练管理器，PPO更新
- `ppo.py`: PPO算法实现
- `train.py`: 训练启动脚本

### 配置文件层级
```
legged_robot_config.py (基础配置)
    ↓ 继承
h1_2_fix.py (H1机器人专用配置)
    ↓ 实例化
H1_2FixCfg (运行时配置对象)
```

---

**总结**: 您当前使用的是最适合仿真研究的配置，使用高度图提供完整地形信息，训练效率高且性能最优。如需真实机器人部署，再考虑启用深度相机模式。

## 🚀 快速开始

### 1. 继续您的训练
```bash
# 继续当前训练（从17500次迭代开始）
python legged_gym/scripts/train.py --task=h1_2 --headless --resume --load_run=Sep09_18-20-06--h1-2

# 测试当前策略
python legged_gym/scripts/play.py --task=h1_2 --load_run=Sep09_18-20-06--h1-2
```

### 2. 常用调试命令
```bash
# 可视化训练（不保存模型）
python legged_gym/scripts/play.py --task=h1_2 --load_run=Sep09_18-20-06--h1-2

# 小规模调试
python legged_gym/scripts/train.py --task=h1_2 --debug --num_envs=64
```

## ⚡ 重要提示

### 🎯 当前状态
- ✅ **训练模式**: 纯教师策略（最优仿真性能）
- ✅ **地形感知**: 高度图（396个采样点）
- ✅ **并行环境**: 4096个
- ✅ **训练进度**: 17500/50000 迭代

### 🔧 关键修改点
1. **奖励调整**: `legged_gym/envs/h1/h1_2_fix.py`
2. **观测修改**: `legged_gym/envs/base/humanoid_robot.py` 第643-802行
3. **动作配置**: `legged_gym/envs/base/legged_robot_config.py` 第172-190行
4. **地形设置**: `challenging_terrain/terrain_base/config.py`

### 💡 优化建议
- 继续训练到50000次迭代以获得最佳性能
- 监控wandb日志中的奖励曲线和成功率
- 如需调整，先修改奖励权重，再重新训练
- 仿真研究无需考虑深度相机模式

---

**📞 需要帮助？**
- 查看wandb训练日志
- 检查 `logs/parkour_new/Sep09_18-20-06--h1-2/` 目录
- 使用 `--debug` 模式进行小规模测试 