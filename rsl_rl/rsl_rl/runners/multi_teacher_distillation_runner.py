# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Multi-Teacher Distillation Runner - Based on rsl_rl_old framework
# 多教师蒸馏运行器 - 基于rsl_rl_old框架

import time
import os
from collections import deque
import statistics
import torch
import warnings

from rsl_rl.algorithms import MultiTeacherDistillation
from rsl_rl.modules import MultiTeacherStudent
from rsl_rl.env import VecEnv
from copy import copy, deepcopy


class MultiTeacherDistillationRunner:
    """多教师蒸馏训练运行器 - 基于rsl_rl_old框架"""

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', 
                 **kwargs):
        """
        多教师蒸馏训练器初始化
        
        Args:
            env: 向量化环境
            train_cfg: 训练配置字典
            log_dir: 日志保存目录
            init_wandb: 是否初始化wandb日志记录
            device: 计算设备
        """
        
        # ========== 解析训练配置 ==========
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        
        # 设置计算设备和环境引用
        self.device = device
        self.env = env

        # ========== 创建多教师学生网络 ==========
        print("创建多教师学生网络...")
        
        # 获取观测和动作维度 - 现在返回完整的731维观测
        obs = env.get_observations()
        # 现在obs是完整的731维张量，学生和教师都使用相同的观测
        actor_obs = obs
        # 修复：确保critic_obs不为None
        privileged_obs = env.get_privileged_observations() if hasattr(env, 'get_privileged_observations') else None
        critic_obs = privileged_obs if privileged_obs is not None else obs

        # 创建多教师学生网络
        actor_critic = MultiTeacherStudent(
            num_prop=env.cfg.env.n_proprio,
            num_scan=env.cfg.env.n_scan,
            num_critic_obs=critic_obs.shape[-1] if hasattr(critic_obs, 'shape') else env.num_obs,
            num_priv_latent=env.cfg.env.n_priv_latent,
            num_priv_explicit=env.cfg.env.n_priv,
            num_hist=env.cfg.env.history_len,
            num_actions=env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        # ========== 创建多教师蒸馏算法 ==========
        self.alg = MultiTeacherDistillation(
            actor_critic,
            device=self.device,
            **self.alg_cfg
        )
        
        # 设置环境引用，用于真实的动作处理
        self.alg.env = self.env

        # ========== 设置训练参数 ==========
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # ========== 初始化经验缓冲区 ==========
        actor_obs_shape = [actor_obs.shape[-1]] if hasattr(actor_obs, 'shape') else [env.num_obs]
        critic_obs_shape = [critic_obs.shape[-1]] if hasattr(critic_obs, 'shape') else [env.num_privileged_obs or env.num_obs]
        
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            actor_obs_shape,
            critic_obs_shape,
            [self.env.num_actions],
        )

        # ========== 初始化日志记录 ==========
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        print(f"多教师蒸馏运行器初始化完成")
        print(f"- 教师数量: {actor_critic.num_teachers}")
        print(f"- 学生网络参数: {sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)}")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        多教师蒸馏训练主循环
        
        Args:
            num_learning_iterations: 训练迭代次数
            init_at_random_ep_len: 是否随机初始化episode长度（未使用）
        """
        
        # ========== 初始化损失记录 ==========
        mean_distillation_loss = 0.
        mean_behavior_cloning_loss = 0.
        mean_diversity_loss = 0.
        mean_total_loss = 0.

        # ========== 初始化日志记录缓冲区 ==========
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        
        # 当前episode累计值
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # ========== 获取初始观测 ==========
        obs = self.env.get_observations()
        if isinstance(obs, dict):
            actor_obs = obs.get("policy", list(obs.values())[0])
            critic_obs = obs.get("critic", obs.get("privileged", list(obs.values())[-1]))
        else:
            actor_obs = obs
            # 修复：确保critic_obs不为None
            privileged_obs = self.env.get_privileged_observations() if hasattr(self.env, 'get_privileged_observations') else None
            critic_obs = privileged_obs if privileged_obs is not None else obs
            
        actor_obs, critic_obs = actor_obs.to(self.device), critic_obs.to(self.device)
        
        # 设置网络为训练模式
        self.alg.train_mode()

        # ========== 设置训练迭代范围 ==========
        tot_iter = self.current_learning_iteration + num_learning_iterations
        start_learning_iteration = copy(self.current_learning_iteration)

        # ========== 主训练循环 ==========
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ========== 收集经验（Rollout）==========
            # 蒸馏训练：收集数据时不需要梯度，但要保存原始观测用于后续训练
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    # 学生策略执行动作（不需要梯度）
                    actions = self.alg.act(actor_obs, critic_obs, {})
                    
                    # 环境步进
                    step_obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    
                    # 获取环境处理后的动作（这是实际使用的动作，与play.py一致）
                    processed_actions = self.env.actions.clone()
                    
                    # 调试：打印动作范围对比
                    if i == 0 and it % 10 == 0:  # 每10个迭代打印一次
                        print(f"[DISTILL] 原始学生动作: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
                        print(f"[DISTILL] 环境处理后动作: [{processed_actions.min().item():.4f}, {processed_actions.max().item():.4f}]")
                        print(f"[DISTILL] 动作是否相同: {torch.allclose(actions, processed_actions)}")
                        if not torch.allclose(actions, processed_actions):
                            print(f"[DISTILL] 动作差异: {(actions - processed_actions).abs().max().item():.6f}")
                    
                    # 更新观测
                    if isinstance(step_obs, dict):
                        actor_obs = step_obs.get("policy", list(step_obs.values())[0])
                        critic_obs = step_obs.get("critic", step_obs.get("privileged", list(step_obs.values())[-1]))
                    else:
                        actor_obs = step_obs
                        critic_obs = privileged_obs if privileged_obs is not None else step_obs
                    
                    # 转移到设备
                    actor_obs, critic_obs = actor_obs.to(self.device), critic_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    
                    # 更新存储的动作为环境处理后的动作（与play.py一致）
                    self.alg.transition.actions = processed_actions.detach()
                    
                    # 处理环境步骤
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    # ========== 日志记录 ==========
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        
                        cur_reward_sum += total_rew
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # ========== 计算回报 ==========
            start = stop
            self.alg.compute_returns(critic_obs)

            # ========== 多教师蒸馏更新 ==========
            loss_values = self.alg.update()
            (mean_total_loss, mean_distillation_loss, mean_behavior_cloning_loss, 
             mean_diversity_loss, _, _, _) = loss_values

            stop = time.time()
            learn_time = stop - start
            
            # ========== 日志记录和模型保存 ==========
            if self.log_dir is not None:
                self.log(locals())
            
            # 模型保存策略
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            
            ep_infos.clear()
        
        # ========== 训练结束保存最终模型 ==========
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        """日志记录函数"""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        
        # episode信息日志
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # 性能指标
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # 蒸馏损失日志
        wandb_dict['Loss/total_loss'] = locs['mean_total_loss']
        wandb_dict['Loss/distillation_loss'] = locs['mean_distillation_loss']
        wandb_dict['Loss/behavior_cloning_loss'] = locs['mean_behavior_cloning_loss']
        wandb_dict['Loss/diversity_loss'] = locs['mean_diversity_loss']

        # 性能日志
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection_time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        
        # 蒸馏特定指标
        if 'student_action_stats' in locs:
            wandb_dict['Student/action_mean'] = locs['student_action_stats'].get('mean', 0)
            wandb_dict['Student/action_std'] = locs['student_action_stats'].get('std', 0)
            wandb_dict['Student/action_min'] = locs['student_action_stats'].get('min', 0)
            wandb_dict['Student/action_max'] = locs['student_action_stats'].get('max', 0)
        
        if 'teacher_action_stats' in locs:
            wandb_dict['Teacher/action_mean'] = locs['teacher_action_stats'].get('mean', 0)
            wandb_dict['Teacher/action_std'] = locs['teacher_action_stats'].get('std', 0)
            wandb_dict['Teacher/action_min'] = locs['teacher_action_stats'].get('min', 0)
            wandb_dict['Teacher/action_max'] = locs['teacher_action_stats'].get('max', 0)
        
        if 'action_difference' in locs:
            wandb_dict['Distillation/student_teacher_diff'] = locs['action_difference']
        
        if 'terrain_distribution' in locs:
            for terrain_id, count in locs['terrain_distribution'].items():
                wandb_dict[f'Terrain/terrain_{terrain_id}_samples'] = count
        
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/reward_std'] = statistics.stdev(locs['rewbuffer']) if len(locs['rewbuffer']) > 1 else 0

        # 使用wandb记录
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(wandb_dict, step=locs['it'])
                print(f"[DEBUG] Wandb日志已记录，step={locs['it']}, metrics={len(wandb_dict)}")
            else:
                print("[WARNING] Wandb run未初始化，跳过日志记录")
        except ImportError:
            pass
        except Exception as e:
            print(f"[ERROR] Wandb日志记录失败: {e}")

        # 控制台输出
        str_header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n"""
                          f"""{'Distillation loss:':>{pad}} {locs['mean_distillation_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_behavior_cloning_loss']:.4f}\n"""
                          f"""{'Diversity loss:':>{pad}} {locs['mean_diversity_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                          f"""{'Total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        
        curr_it = locs['it'] - locs.get('start_learning_iteration', 0)
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it) if curr_it > 0 else 0
        mins = int(eta // 60)
        secs = eta % 60
        
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        """保存模型"""
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        torch.save(state_dict, path)
        print(f"模型已保存至: {path}")

    def load(self, path, load_optimizer=True):
        """加载模型"""
        print("*" * 80)
        print("从 {} 加载模型...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """获取推理策略"""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference