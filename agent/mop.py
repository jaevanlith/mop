import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.pbrl import Actor, ActorND, Critic

class ActorMT(Actor):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1):
        super().__init__(state_dim, action_dim, max_action)
        
        # Add an extra input feature for task id
        self.l1 = nn.Linear(state_dim + 1, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, task_id, analyse=False):
        # Concatenate state and task id
        task_id_tensor = torch.tensor(task_id, dtype=torch.float32, device=state.device).unsqueeze(0).expand(state.shape[0], -1)
        input = torch.cat((state, task_id_tensor), 1)
        # Forward pass
        l1_out = F.relu(self.l1(input))
        l2_out = F.relu(self.l2(l1_out))
        l3_out = torch.tanh(self.l3(l2_out))
        a = self.max_action * l3_out
        
        if analyse:
            return a, l1_out, l2_out, l3_out
        else:
            return a


class ActorMTND(ActorND):
    def __init__(self, state_dim, action_dim, max_action=1):
        super().__init__(state_dim, action_dim, max_action)

        # Add an extra input feature for task id
        self.l1 = nn.Linear(state_dim + 1, 256)

    def forward(self, state, task_id, suff_stats=False):
        # Concatenate state and task id
        task_id_tensor = torch.tensor(task_id, dtype=torch.float32, device=state.device).unsqueeze(0).expand(state.shape[0], -1)
        input = torch.cat((state, task_id_tensor), 1)
        # Forward pass
        a = F.relu(self.l1(input))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        # Separate mean and log std dev
        mean, log_std = a[:, :self.action_dim], a[:, self.action_dim:]

        # Either return sufficient statistics or sampled action
        if suff_stats:
            return mean, log_std
        else:
            return self.sample_action(mean, log_std)


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
                 hidden_dim,
                 device,
                 lr,
                 ensemble,
                 deterministic_actor=True):
        self.device = device
        self.lr = lr
        self.ensemble = ensemble
        self.max_action = 1.0

        # Init multi-task actor and its optimizer
        self.deterministic_actor = deterministic_actor
        if self.deterministic_actor:
            self.actor = ActorMT(state_dim, action_dim, hidden_dim, self.max_action).to(device)
        else:
            self.actor = ActorMTND(state_dim, action_dim, self.max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Initialize ensemble of critics and their optimizers
        self.critic, self.critic_opt = [], []
        for _ in range(self.ensemble):
            single_critic = CriticMT(state_dim, action_dim).to(device)
            single_critic_opt = torch.optim.Adam(single_critic.parameters(), lr=lr)
            
            self.critic.append(single_critic)
            self.critic_opt.append(single_critic_opt)

        # Init loss function
        if self.deterministic_actor:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = self.kl_div

    
    def kl_div(mu1, log_std1, mu2, log_std2):
        # Compute variances
        var1 = torch.exp(log_std1) ** 2
        var2 = torch.exp(log_std2) ** 2

        # Convert to diagional covariance matrices
        cov1 = torch.diag_embed(var1)
        cov2 = torch.diag_embed(var2)
        cov2_inverse = torch.linalg.inv(cov2)

        # Compute trace(cov2_inverse @ cov1)
        tr = cov2_inverse.matmul(cov1).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        # Compute dimensionality
        k = mu1.size(-1)
            
        # Compute KL divergence
        kl = 0.5 * (tr - k +
                    (mu2-mu1).unsqueeze(dim=1).matmul(cov2_inverse).matmul((mu2-mu1).unsqueeze(dim=1).transpose(-1, -2)).squeeze(dim=-1).squeeze(dim=-1) +
                    torch.log(torch.linalg.det(cov2) / torch.linalg.det(cov1)))

        return kl


    def act(self, state, task_id):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return self.actor(state, task_id).cpu().numpy()[0]
    
    
    def infer_analysis(self, state, task_id):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, l1_out, l2_out, l3_out = self.actor(state, task_id, analyse=True)
        return a.cpu().numpy()[0], l1_out.cpu().numpy()[0], l2_out.cpu().numpy()[0], l3_out.cpu().numpy()[0]


    def update_a2a(self, task_id, teacher, state):
        # Compute loss
        if self.deterministic_actor:
            actor_loss = self.criterion(self.actor(state, task_id), teacher.actor(state).detach())
        else:
            mu, log_std = self.actor(state, task_id, suff_stats=True)
            mu_teacher, log_std_teacher = teacher.actor(state, suff_stats=True)
            actor_loss = self.criterion(mu, log_std, mu_teacher, log_std_teacher)

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


    def load(self, directory):
        # Load actor
        self.actor.load_state_dict(torch.load(directory / "actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load(directory / "actor_opt.pt"))
