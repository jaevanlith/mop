from agent.pbrl import PBRLAgent
from agent.mop import ActorMT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils

class PBRLAgent_MT(PBRLAgent):
    def __init__(self,
				 name,
				 obs_shape,
				 action_shape,
				 device,
				 lr,
				 hidden_dim,
				 critic_target_tau,
				 actor_target_tau,
				 policy_freq,
	             policy_noise,
	             noise_clip,
				 use_tb,
				 # alpha,
				 batch_size,
				 num_expl_steps,
	             # PBRL parameters
	             num_random,
	             ucb_ratio_in,
	             ucb_ratio_ood_init,
	             ucb_ratio_ood_min,
	             ood_decay_factor,
	             ensemble,
	             ood_noise,
	             share_ratio,
                 task_id,
	             has_next_action=False):
        # Initialize from PBRLAgent
        super().__init__(name,
				 obs_shape,
				 action_shape,
				 device,
				 lr,
				 hidden_dim,
				 critic_target_tau,
				 actor_target_tau,
				 policy_freq,
	             policy_noise,
	             noise_clip,
				 use_tb,
				 # alpha,
				 batch_size,
				 num_expl_steps,
	             # PBRL parameters
	             num_random,
	             ucb_ratio_in,
	             ucb_ratio_ood_init,
	             ucb_ratio_ood_min,
	             ood_decay_factor,
	             ensemble,
	             ood_noise,
	             share_ratio,
	             has_next_action=False)
        
        # Init multi-task actor and its optimizer
        self.actor = ActorMT(obs_shape[0], action_shape[0]).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        print("MT Actor parameters:", utils.total_parameters(self.actor))
        # Activate training mode
        self.train()
        # Agent is concerned with a particular task, actor is copied across them
        self.task_id = task_id


    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action = self.actor(obs, self.task_id)
        if step < self.num_expl_steps:
            action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def update_critic(self, obs, action, reward, discount, next_obs, step, total_step, bool_flag):
        self.share_ratio_now = utils.decay_linear(t=step, init=self.share_ratio, minimum=1.0, total_steps=total_step // 2)

        metrics = dict()
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, self.task_id) + noise).clamp(-self.max_action, self.max_action)

            # ood sample 1
            sampled_current_actions = self.actor(obs, self.task_id).unsqueeze(1).repeat(1, self.num_random, 1).view(
                        action.shape[0]*self.num_random, action.shape[1])
            noise_current = (torch.randn_like(sampled_current_actions) * self.ood_noise).clamp(-self.noise_clip, self.noise_clip)
            sampled_current_actions = (sampled_current_actions + noise_current).clamp(-self.max_action, self.max_action)

            # ood sample 2
            sampled_next_actions = self.actor(next_obs, self.task_id).unsqueeze(1).repeat(1, self.num_random, 1).view(
                        action.shape[0]*self.num_random, action.shape[1])
            noise_next = (torch.randn_like(sampled_next_actions) * self.ood_noise).clamp(-self.noise_clip, self.noise_clip)
            sampled_next_actions = (sampled_next_actions + noise_next).clamp(-self.max_action, self.max_action)

            # random sample
            random_actions = torch.FloatTensor(action.shape[0]*self.num_random, action.shape[1]).uniform_(
                        -self.max_action, self.max_action).to(self.device)

        # TODO: UCB and Q-values
        ucb_current, q_pred = self.ucb_func(obs, action)     # (1024,1).  lenth=ensemble, q_pred[0].shape=(1024,1)
        ucb_next, target_q_pred = self.ucb_func_target(next_obs, next_action)   # (1024,1).  lenth=ensemble, target_q_pred[0].shape=(1024,1)

        ucb_curr_actions_ood, qf_curr_actions_all_ood = self.ucb_func(obs, sampled_current_actions)      # (1024*num_random, 1), length=ensemble, (1024*num_random, 1)
        ucb_next_actions_ood, qf_next_actions_all_ood = self.ucb_func(next_obs, sampled_next_actions)    # 同上
        # ucb_rand_ood, qf_rand_actions_all_ood = self.ucb_func(obs, random_actions)

        for qf_index in np.arange(self.ensemble):
            ucb_ratio_in_flag = bool_flag * self.ucb_ratio_in + (1 - bool_flag) * self.ucb_ratio_in * self.share_ratio_now
            ucb_ratio_in_flag = np.expand_dims(ucb_ratio_in_flag, 1)
            q_target = reward + discount * (target_q_pred[qf_index] - torch.from_numpy(ucb_ratio_in_flag.astype(np.float32)).cuda() * ucb_next)  # (1024, 1), (1024, 1), (1024, 1)
            # print("bool flag", bool_flag[:10],  bool_flag[-10:])
            # print("ucb_ratio_in_flag", q_target.shape, ucb_ratio_in_flag.shape, ucb_next.shape, (torch.from_numpy(ucb_ratio_in_flag.astype(np.float32)).cuda() * ucb_next).shape, ucb_ratio_in_flag[:10])

            # q_target = reward + discount * (target_q_pred[qf_index] - self.ucb_ratio_in * ucb_next)  # (1024, 1), (1024, 1), (1024, 1)
            q_target = q_target.detach()
            qf_loss_in = F.mse_loss(q_pred[qf_index], q_target)

            # TODO: ood loss
            cat_qf_ood = torch.cat([qf_curr_actions_all_ood[qf_index],
                                    qf_next_actions_all_ood[qf_index]], 0)
            # assert cat_qf_ood.size() == (1024*self.num_random*3, 1)

            ucb_ratio_ood_flag = bool_flag * self.ucb_ratio_ood + (1 - bool_flag) * self.ucb_ratio_ood * self.share_ratio_now
            ucb_ratio_ood_flag = np.expand_dims(ucb_ratio_ood_flag, 1).repeat(self.num_random, axis=1).reshape(-1, 1).astype(np.float32)
            # print("ucb_ratio_ood_flag 1", ucb_ratio_ood_flag.shape, ucb_curr_actions_ood.shape)

            cat_qf_ood_target = torch.cat([
                torch.maximum(qf_curr_actions_all_ood[qf_index] - torch.from_numpy(ucb_ratio_ood_flag).cuda() * ucb_curr_actions_ood, torch.zeros(1).cuda()),
                torch.maximum(qf_next_actions_all_ood[qf_index] - torch.from_numpy(ucb_ratio_ood_flag).cuda() * ucb_next_actions_ood, torch.zeros(1).cuda())], 0)
            # print("ucb_ratio_ood_flag 2", cat_qf_ood_target.shape, qf_curr_actions_all_ood[qf_index].shape)
            cat_qf_ood_target = cat_qf_ood_target.detach()

            # assert cat_qf_ood_target.size() == (1024*self.num_random*3, 1)
            qf_loss_ood = F.mse_loss(cat_qf_ood, cat_qf_ood_target)
            critic_loss = qf_loss_in + qf_loss_ood

            # Update the Q-functions
            self.critic_opt[qf_index].zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_opt[qf_index].step()

        # change the ood ratio
        self.ucb_ratio_ood = max(self.ucb_ratio_ood_init * self.ood_decay_factor ** step, self.ucb_ratio_ood_min)

        if self.use_tb:
            metrics['critic_target_q'] = q_target.mean().item()
            metrics['critic_q1'] = q_pred[0].mean().item()
            # metrics['critic_q2'] = q_pred[1].mean().item()
            # ucb
            metrics['ucb_current'] = ucb_current.mean().item()
            metrics['ucb_next'] = ucb_next.mean().item()
            metrics['ucb_curr_actions_ood'] = ucb_curr_actions_ood.mean().item()
            metrics['ucb_next_actions_ood'] = ucb_next_actions_ood.mean().item()
            # loss
            metrics['critic_loss_in'] = qf_loss_in.item()
            metrics['critic_loss_ood'] = qf_loss_ood.item()
            metrics['ucb_ratio_ood'] = self.ucb_ratio_ood
            metrics['share_ratio_now'] = self.share_ratio_now
        return metrics
    

    def update_actor(self, obs, action):
        metrics = dict()

        # Compute actor loss
        pi = self.actor(obs, self.task_id)

        Qvalues = []
        for i in range(self.ensemble):
            Qvalues.append(self.critic[i](obs, pi))           # (1024, 1)
        Qvalues_min = torch.min(torch.hstack(Qvalues), dim=1, keepdim=True).values
        assert Qvalues_min.size() == (1024, 1)

        actor_loss = -1. * Qvalues_min.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics