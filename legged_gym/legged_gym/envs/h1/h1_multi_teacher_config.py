# Multi-Teacher Distillation Configuration for H1 Robot
# 多教师蒸馏训练配置 - H1机器人

from legged_gym.envs.h1.h1_2_fix import H1_2FixCfgPPO, H1_2FixCfg

class H1_2FixCfgMultiTeacher(H1_2FixCfg):
    """H1_2机器人多教师蒸馏环境配置"""
    
    class env(H1_2FixCfg.env):
        """确保观测维度与教师模型匹配"""
        # 继承父类的所有配置，确保观测维度计算一致
        # num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv = 731
        pass
    
    # 观测组配置 - 定义不同观测组的结构
    obs_groups = {
        "policy": ["policy"],           # 学生策略观测（公开信息）
        "critic": ["critic"],           # Critic观测（包含特权信息）  
        "teacher": ["teacher"],         # 教师策略观测（包含特权信息）
        "terrain_info": ["terrain_info"] # 地形信息（用于教师选择）
    }

class H1_2FixCfgDistillPPO(H1_2FixCfgPPO):
    """H1_2机器人多教师蒸馏训练配置 - 修复类名冲突"""
    
    class policy:
        """多教师学生策略配置"""
        class_name = "MultiTeacherStudent"
        
        # 教师模型数量和路径 - 使用实际存在的模型文件
        num_teachers = 5
        teacher_model_paths = [
            "/home/asus/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/hurdle1/model_50000.pt",
            "/home/asus/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/hurdle1/model_50000.pt",  # 复用第一个教师
            "/home/asus/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/hurdle1/model_50000.pt",  # 复用第一个教师
            "/home/asus/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/hurdle1/model_50000.pt",  # 复用第一个教师
            "/home/asus/Humanoid-Terrain-Bench-kelun/legged_gym/logs/h1_distillation/hurdle1/model_50000.pt",  # 复用第一个教师
        ]
        
        # 网络配置 - 匹配教师模型的实际结构
        student_obs_normalization = False
        teacher_obs_normalization = False
        actor_hidden_dims = [512, 256, 128]  # 匹配教师模型结构
        critic_hidden_dims = [512, 256, 128] # 匹配教师模型结构
        teacher_hidden_dims = [512, 256, 128] # 匹配教师模型结构
        scan_encoder_dims = [128, 64, 32]     # 匹配教师模型结构
        activation = "elu"
        init_noise_std = 1.0
        # 兼容rsl_rl_old的参数 - 匹配教师模型的priv_encoder结构
        priv_encoder_dims = [64, 20]  # 匹配教师模型: [64, 20]
        tanh_encoder_output = False
    
    class algorithm:
        """多教师蒸馏算法配置"""
        class_name = "MultiTeacherDistillation"
        
        # 训练参数
        num_learning_epochs = 3
        num_mini_batches = 4
        learning_rate = 1e-4
        max_grad_norm = 1.0
        # 蒸馏损失配置
        distillation_loss_coef = 1.0
        behavior_cloning_coef = 1.0  # 主要损失：模仿教师
        diversity_loss_coef = 0.1    # 辅助损失：鼓励地形专门化
        # 地形自适应权重
        terrain_adaptive_weights = True  # 启用地形自适应
        weight_temperature = 1.0
    
    class runner:
        """蒸馏训练运行器配置"""
        class_name = 'MultiTeacherDistillationRunner'  # 关键：指定使用多教师蒸馏训练器
        run_name = ''
        experiment_name = 'h1_2_distill' 
        max_iterations = 10000
        save_interval = 500
        resume = False
        load_run = -1
        checkpoint = -1
        
        # 添加必需的训练参数
        num_steps_per_env = 24
        seed = 1
        logger = "wandb"  # 使用wandb日志
        
        # 添加wandb配置
        wandb_project = "h1_distillation"  # wandb项目名称
        
        # 观测组配置 - 修复：使用HumanoidRobot环境支持的观测组
        obs_groups = {
            "policy": ["policy"],           # 学生策略观测（公开信息）
            "critic": ["critic"],           # Critic观测（包含特权信息）
            "teacher": ["teacher"],         # 教师观测（包含特权信息）
            "terrain_info": ["terrain_info"] # 地形信息（用于教师选择）
        }


# 导出配置类
__all__ = ["H1_2FixCfgMultiTeacher", "H1_2FixCfgDistillPPO"]