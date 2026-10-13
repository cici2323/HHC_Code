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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import numpy as np
from rsl_rl.algorithms import PPOMimicPriv
from rsl_rl.modules import *
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
from rsl_rl.utils.utils import Normalizer
from rsl_rl.residual.checkpoints import load_checkpoint

class OnPolicyRunnerMimicPriv:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        if "distill" in train_cfg:
            self.if_distill = True
            self.distill_cfg = train_cfg["distill"]
        else:
            self.if_distill = False
            self.distill_cfg = None
        self.device = device
        self.init_wandb = bool(init_wandb)
        self.pipeline_metadata = {}
        self.env = env
        self.dof_names = env.dof_names

        policy_class = eval(self.cfg["policy_class_name"])
        actor_critic = policy_class(self.env.cfg.env.n_proprio,  self.env.cfg.env.n_teacher_priv,
                                                      self.env.cfg.env.n_demo,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.n_priv,
                                                      self.env.cfg.env.history_len,
                                                      self.env.cfg.env.num_policy_actions,
                                                      **self.policy_cfg).to(self.device)
        print(actor_critic)
        estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                    self.policy_cfg["scan_encoder_dims"][-1], 
                                                    self.depth_encoder_cfg["hidden_dims"],
                                                    )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None
        
        if self.if_distill:
            student_actor = ActorDistillPriv(actor_critic.actor, self.distill_cfg).to(self.device)
        else:
            student_actor = None
        
        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPOMimicPriv = alg_class(self.env, 
                                  actor_critic, 
                                  estimator, self.estimator_cfg, 
                                  depth_encoder, self.depth_encoder_cfg, depth_actor, student_actor, self.distill_cfg,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.cfg.env.num_policy_actions],
        )

        self.learn = self.learn_RL if not self.if_distill else self.learn_distill
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._teacher_initialized = not self.if_distill
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # AMP
                        task_rew_buf.extend(cur_task_rew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_task_rew_sum[new_ids] = 0
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, grad_penalty_coef  = self.alg.update(clear_storage=not hist_encoding)
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            self.current_learning_iteration = it + 1
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_distill(self, num_learning_iterations, init_at_random_ep_len=False):
        if not self._teacher_initialized:
            raise RuntimeError("Call load_teacher(teacher_checkpoint) before distillation, or load(student_checkpoint) to resume")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations().to(self.device)
        infos = {}
        infos["extra_hist_obs"] = self.env.get_extra_hist_obs().to(self.device)
        self.alg.actor_critic.eval()
        self.alg.estimator.eval()
        self.alg.student_actor.eval()
        self.alg.student_actor.student_actor_backbone.train()

        buffer_size = self.distill_cfg["num_steps_per_env"] * self.env.num_envs
        obs_teacher_hist_buffer = ReplayBuffer(obs.shape[1]+ infos["extra_hist_obs"].shape[1], buffer_size, self.device)
        
        num_pretrain_iter = self.distill_cfg["num_pretrain_iter"]
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # actions_teacher_buffer = []
            # actions_student_buffer = []
            for i in range(self.distill_cfg["num_steps_per_env"]):
                
                with torch.no_grad():
                    policy_obs = self.alg.estimated_observations(obs)
                    actions_teacher = self.alg.student_actor.act_teacher(policy_obs, hist_encoding=True)
                    # actions_teacher_buffer.append(actions_teacher)

                obs_plus_hist = torch.cat([policy_obs, infos["extra_hist_obs"].to(self.device)], dim=1)
                obs_teacher_hist_buffer.insert(obs_plus_hist)

                obs_student = obs_plus_hist[:, :obs.shape[1]].clone()
                obs_hist = obs_plus_hist[:, obs.shape[1]:].clone()
                with torch.no_grad():
                    actions_student = self.alg.student_actor(obs_student, obs_hist, hist_encoding=True)
                # actions_student_buffer.append(actions_student)

                if it < num_pretrain_iter:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_teacher.detach())  # obs has changed to next_obs !! if done obs has been reset
                else:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())  # obs has changed to next_obs !! if done obs has been reset
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
            stop = time.time()
            collection_time = stop - start
            start = stop

            # actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            # actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            
            teacher_obs_generater = obs_teacher_hist_buffer.feed_forward_generator(self.distill_cfg["num_mini_batches"], buffer_size // self.distill_cfg["num_mini_batches"])
            # extra_hist_obs_generater = extra_hist_obs_buffer.feed_forward_generator(self.distill_cfg["num_mini_batches"], buffer_size // self.distill_cfg["num_mini_batches"])
            losses = []
            # for (teacher_obs, extra_hist_obs) in zip(teacher_obs_generater, extra_hist_obs_generater):
            for obs_plus_hist in teacher_obs_generater:
                with torch.no_grad():
                    teacher_obs = obs_plus_hist[:, :obs.shape[1]]
                    actions_teacher_buffer = self.alg.student_actor.act_teacher(teacher_obs, hist_encoding=True)
                obs_hist = obs_plus_hist[:, obs.shape[1]:].clone()
                actions_student_buffer = self.alg.student_actor(teacher_obs, obs_hist, hist_encoding=True)
                distill_loss = self.alg.update_distill(actions_student_buffer, actions_teacher_buffer)
                losses.append(distill_loss)
            distill_loss = np.mean(losses)
            
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log_distill(locals())
            self.current_learning_iteration = it + 1
            self.alg.counter = self.current_learning_iteration
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log_distill(self, locs, width=80, pad=35):
        self.tot_timesteps += self.distill_cfg["num_steps_per_env"] * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "tracking" in key:
                    wandb_dict['Episode_rew_tracking/' + key] = value
                else:
                    wandb_dict['Episode_rew_regularization/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_distill/student_actor'] = locs['distill_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        if self.init_wandb and wandb.run is not None:
            wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it'] + 1}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Student actor loss:':>{pad}} {locs['distill_loss']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb_dict['Episode_rew/' + key] = value
                if "tracking" in key:
                    wandb_dict['Episode_rew_tracking/' + key] = value
                elif "curriculum" in key:
                    wandb_dict['Episode_curriculum/' + key] = value
                else:
                    wandb_dict['Episode_rew_regularization/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        std_list = self.alg.actor_critic.std.tolist()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_func'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/grad_penalty_loss'] = locs['mean_grad_penalty_loss']
        wandb_dict['Loss/grad_penalty_coef'] = locs['grad_penalty_coef']
        # wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        # wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Adaptation/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']


        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        for i, std in enumerate(std_list):
            wandb_dict[f'Policy/std_{self.dof_names[i]}'] = std
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            # wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = statistics.mean(locs['task_rew_buf'])
            # wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        if self.init_wandb and wandb.run is not None:
            wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it'] + 1}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""

                          f"""{'Grad penalty loss:':>{pad}} {locs['mean_grad_penalty_loss']:.4f}\n"""

                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['task_rew_buf']):.2f}\n"""
                        #   f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                        #   f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n"""
                       f"""{'Grad_penalty_coef:':>{pad}} {locs['grad_penalty_coef']:.4f}\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'pipeline_checkpoint_version': 2,
            'pipeline_metadata': self.pipeline_metadata,
            'stage': 'student' if self.if_distill else 'teacher',
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.estimator_optimizer.state_dict(),
            'hist_encoder_optimizer_state_dict': self.alg.hist_encoder_optimizer.state_dict(),
            'iter': self.current_learning_iteration,  # next iteration to execute
            'algorithm_counter': self.alg.counter,
            'learning_rate': self.alg.learning_rate,
            'tot_timesteps': self.tot_timesteps,
            'tot_time': self.tot_time,
            'infos': infos,
        }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
            state_dict['depth_encoder_optimizer_state_dict'] = self.alg.depth_encoder_optimizer.state_dict()
            state_dict['depth_actor_optimizer_state_dict'] = self.alg.depth_actor_optimizer.state_dict()
        if self.if_distill:
            state_dict['student_actor_state_dict'] = self.alg.student_actor.state_dict()
            state_dict['student_actor_optimizer_state_dict'] = self.alg.student_actor_optimizer.state_dict()
        torch.save(state_dict, path)

    def _load_model_weights(self, checkpoint):
        self.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.alg.estimator.load_state_dict(checkpoint['estimator_state_dict'], strict=True)
        self.alg._constrain_std()

    def load_teacher(self, path):
        """Initialize a new student stage from teacher weights, without resume state."""
        if not self.if_distill:
            raise ValueError("load_teacher is for initializing student distillation; use load for teacher resume")
        checkpoint = load_checkpoint(path, self.device)
        if checkpoint.get('stage', 'teacher') != 'teacher' or 'student_actor_state_dict' in checkpoint:
            raise ValueError("load_teacher requires a teacher checkpoint; use load to resume a student")
        self._load_model_weights(checkpoint)
        # ActorDistillPriv shares this actor, so this also updates its teacher.
        self.alg.student_actor._orig_actor.load_state_dict(self.alg.actor_critic.actor.state_dict(), strict=True)
        self.current_learning_iteration = 0
        self.alg.counter = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        for optimizer in (self.alg.optimizer, self.alg.estimator_optimizer,
                          self.alg.hist_encoder_optimizer, self.alg.student_actor_optimizer):
            optimizer.state.clear()
        self._teacher_initialized = True
        return checkpoint.get('infos')

    def load(self, path, load_optimizer=True):
        """Strict same-stage resume; teacher-only input cannot silently start a student."""
        checkpoint = load_checkpoint(path, self.device)
        if self.pipeline_metadata and checkpoint.get('pipeline_metadata') != self.pipeline_metadata:
            raise ValueError("Checkpoint pipeline_metadata differs from the current task/reward configuration")
        expected_stage = 'student' if self.if_distill else 'teacher'
        if checkpoint.get('stage', expected_stage) != expected_stage:
            raise ValueError(f"Expected a {expected_stage} checkpoint; use load_teacher to initialize distillation")
        if self.if_distill and 'student_actor_state_dict' not in checkpoint:
            raise ValueError("Student resume requires student_actor_state_dict; use load_teacher for a teacher checkpoint")
        optimizers = {
            'optimizer_state_dict': self.alg.optimizer,
            'estimator_optimizer_state_dict': self.alg.estimator_optimizer,
            'hist_encoder_optimizer_state_dict': self.alg.hist_encoder_optimizer,
        }
        if self.if_distill:
            optimizers['student_actor_optimizer_state_dict'] = self.alg.student_actor_optimizer
        if self.if_depth:
            optimizers['depth_encoder_optimizer_state_dict'] = self.alg.depth_encoder_optimizer
            optimizers['depth_actor_optimizer_state_dict'] = self.alg.depth_actor_optimizer
        if load_optimizer:
            required = set(optimizers) if checkpoint.get('pipeline_checkpoint_version', 0) >= 2 else {'optimizer_state_dict'}
            if self.if_distill:
                required.add('student_actor_optimizer_state_dict')
            missing = required - checkpoint.keys()
            if missing:
                raise ValueError(f"Checkpoint cannot resume optimizer state: missing {sorted(missing)}")
        self._load_model_weights(checkpoint)
        if self.if_distill:
            self.alg.student_actor.load_state_dict(checkpoint['student_actor_state_dict'], strict=True)
        if self.if_depth:
            self.alg.depth_encoder.load_state_dict(checkpoint['depth_encoder_state_dict'], strict=True)
            self.alg.depth_actor.load_state_dict(checkpoint['depth_actor_state_dict'], strict=True)
        if load_optimizer:
            for name, optimizer in optimizers.items():
                if name in checkpoint:
                    optimizer.load_state_dict(checkpoint[name])
                else:
                    warnings.warn(f"Legacy teacher checkpoint lacks {name}; this optimizer starts fresh")
        self.current_learning_iteration = int(checkpoint.get('iter', 0))
        self.alg.counter = int(checkpoint.get('algorithm_counter', self.current_learning_iteration))
        self.alg.learning_rate = float(checkpoint.get('learning_rate', self.alg.optimizer.param_groups[0]['lr']))
        self.tot_timesteps = int(checkpoint.get('tot_timesteps', 0))
        self.tot_time = float(checkpoint.get('tot_time', 0))
        self._teacher_initialized = True
        return checkpoint.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder
    
    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
