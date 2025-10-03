"""
导出蒸馏学生模型为JIT格式
导出后可以用原始的play.py或evaluate.py进行测试
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import torch
import torch.nn as nn

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

class StudentPolicyWrapper(nn.Module):
    """
    学生策略包装器
    将MultiTeacherStudent中的学生网络提取出来，用于独立推理
    """
    def __init__(self, student_network):
        super().__init__()
        self.student = student_network
        
    def forward(self, obs):
        """
        前向传播
        
        Args:
            obs: 观测张量 [batch_size, obs_dim]
            
        Returns:
            actions: 动作张量 [batch_size, action_dim]
        """
        return self.student(obs)

def export_student_jit(args):
    """
    导出学生模型为JIT格式
    
    Args:
        args: 命令行参数
    """
    print("🚀 开始导出学生模型...")
    
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 创建环境（用于获取观测维度）
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 加载蒸馏模型
    train_cfg.runner.resume = True
    distill_runner, _, _ = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args, 
        train_cfg=train_cfg
    )
    
    # 获取MultiTeacherStudent网络
    multi_teacher_student = distill_runner.alg.policy
    
    # 提取学生网络
    student_network = multi_teacher_student.student
    
    # 创建包装器
    student_wrapper = StudentPolicyWrapper(student_network)
    student_wrapper.eval()
    
    # 获取示例输入（学生观测维度）
    obs = env.get_observations()
    if isinstance(obs, dict):
        example_input = obs["policy"][:1]  # 取第一个环境的学生观测
    else:
        # 如果是张量，需要截取学生部分
        # 根据obs_groups配置计算学生观测维度
        student_obs_dim = 185  # proprio(53) + height_scan(132)
        example_input = obs[:1, :student_obs_dim]
    
    print(f"📊 学生观测维度: {example_input.shape}")
    print(f"🎯 动作维度: {student_network.output_dim}")
    
    # 测试前向传播
    with torch.no_grad():
        test_output = student_wrapper(example_input)
        print(f"✅ 测试输出形状: {test_output.shape}")
    
    # 导出为JIT模型
    print("📦 导出JIT模型...")
    traced_student = torch.jit.trace(student_wrapper, example_input)
    
    # 保存路径
    save_dir = os.path.join(distill_runner.log_dir, "exported")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "student_policy.pt")
    
    # 保存模型
    traced_student.save(save_path)
    
    print(f"✅ 学生模型已导出到: {save_path}")
    print(f"📝 使用方法:")
    print(f"   policy = torch.jit.load('{save_path}')")
    print(f"   actions = policy(obs[:, :{example_input.shape[1]}])")
    
    return save_path

if __name__ == '__main__':
    args = get_args()
    export_student_jit(args) 