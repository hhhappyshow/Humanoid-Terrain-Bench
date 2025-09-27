# Python
import torch

# 假设 4 个并行环境，每个环境有 5 个目标，每个目标维度为 3 (x,y,z)
env_goals = torch.arange(4*5*3, dtype=torch.float).view(4,5,3)

# 每个环境当前的目标索引
cur_goal_idx = torch.tensor([0, 2, 4, 1], dtype=torch.long)   # shape: (4,)

# 模拟类方法的行为（future = 0）
indices = (cur_goal_idx[:, None, None] + 0).expand(-1, -1, env_goals.shape[-1])  # shape: (4,1,3)
out = env_goals.gather(1, indices).squeeze(1)  # shape: (4,3)
print("env_goals.shape:", env_goals.shape)
print("cur_goal_idx:", cur_goal_idx)
print("indices.shape:", indices.shape)
print("indices:", indices)
print("out.shape:", out.shape)
print(out)