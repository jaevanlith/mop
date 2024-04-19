import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.pbrl import Actor, Critic
import copy


class ActorMT(Actor):
    def __init__(self, state_dim, action_dim, max_action=1):
        super().__init__(state_dim, action_dim, max_action)
        
        # Add an extra input feature for task id
        self.l1 = nn.Linear(state_dim + 1, 256)

    def forward(self, state, task_id):
        # Concatenate state and task id
        task_id_tensor = torch.tensor(task_id, dtype=torch.float32, device=state.device).unsqueeze(0).expand(state.shape[0], -1)
        input = torch.cat((state, task_id_tensor), 1)
        # Forward pass
        a = F.relu(self.l1(input))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    

class CriticMT(Critic):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

        # Add an extra input feature for task id
        self.l1 = nn.Linear(state_dim + action_dim + 1, 256)

    def forward(self, state, action, task_id):
        # Concatenate state, action, and task id
        task_id_tensor = torch.tensor(task_id, dtype=torch.float32, device=state.device).unsqueeze(0).expand(state.shape[0], -1)
        input = torch.cat([state, action, task_id_tensor], 1)
        # Forward pass
        q1 = F.relu(self.l1(input))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class MOP:
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 device,
                 lr,
                 ensemble):
        self.device = device
        self.lr = lr
        self.ensemble = ensemble
        self.max_action = 1.0

        # Init multi-task actor and its optimizer
        self.actor = ActorMT(state_dim, action_dim, self.max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Initialize ensemble of critics and their optimizers
        self.critic, self.critic_opt = [], []
        for _ in range(self.ensemble):
            single_critic = CriticMT(state_dim, action_dim).to(device)
            single_critic_opt = torch.optim.Adam(single_critic.parameters(), lr=lr)
            
            self.critic.append(single_critic)
            self.critic_opt.append(single_critic_opt)

        # Init loss function
        self.criterion = nn.MSELoss()


    def act(self, state, task_id):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return self.actor(state, task_id).cpu().numpy()[0]


    def update_a2a(self, task_id, teacher, state):
        # Compute loss
        actor_loss = self.criterion(self.actor(state, task_id), teacher.actor(state).detach())

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Return loss
        return actor_loss.item()
    

    def update_c2a(self, task_id, teacher, state):
        # Compute action
        action = self.actor(state, task_id)

        # Compute Q-values
        Qvalues = []
        for i in range(self.ensemble):
            Qvalues.append(teacher.critic[i](state, action))
        Qvalues_min = torch.min(torch.hstack(Qvalues), dim=1, keepdim=True).values
        assert Qvalues_min.size() == (1024, 1)
        # Compute loss
        actor_loss = -1. * Qvalues_min.mean()

        # Optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Return loss
        return actor_loss.item()
    

    def update(self, task_id, teacher, replay_iter, mode="a2a"):
        # Sample from replay buffer
        batch = next(replay_iter)
        state, action, reward, discount, next_obs, bool_flag = utils.to_torch(batch, self.device)

        if mode == "a2a":
            return self.update_a2a(task_id, teacher, state)
        elif mode == "c2a":
            return self.update_c2a(task_id, teacher, state)
        else:
            raise ValueError("Invalid mode. Choose between 'actor' and 'critic'.")
    

    def save(self, directory):
        # Create directory if it doesn't exist
        directory /= "models"
        directory.mkdir(parents=True, exist_ok=True)

        # Save actor
        torch.save(self.actor.state_dict(), directory / "actor.pt")
        torch.save(self.actor_optimizer.state_dict(), directory / "actor_opt.pt")
