import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.pbrl import Actor


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

class MOP:
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 device,
                 lr):
        self.lr = lr
        self.device = device
        self.max_action = 1.0
        self.use_tb = True

        # Init multi-task actor and its optimizer
        self.actor = ActorMT(state_dim, action_dim, self.max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Init loss function
        self.criterion = nn.MSELoss()


    def act(self, state, task_id):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return self.actor(state, task_id).cpu().numpy()[0]


    def update_actor(self, task_id, teacher, replay_iter):
        # Sample from replay buffer
        batch = next(replay_iter)
        state, action, reward, discount, next_obs, bool_flag = utils.to_torch(batch, self.device)
        bool_flag = bool_flag.cpu().detach().numpy()

        # Compute loss
        actor_loss = self.criterion(self.actor(state, task_id), teacher.actor(state).detach())

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Log metrics
        metrics = dict()
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
        
        return metrics
