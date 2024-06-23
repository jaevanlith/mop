import hydra
from pathlib import Path
from replay_buffer import make_replay_loader
import dmc
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
import numpy as np

def estimate_distribution(states):
    states_transpose = states.cpu().numpy().T
    kde = gaussian_kde(states_transpose)
    return kde

def kl_divergence(cfg, offline_replay_iter, online_replay_iter):
    offline_states = next(offline_replay_iter)[0]
    online_states = next(online_replay_iter)[0]
    
    assert offline_states.shape == online_states.shape

    print('Offline state distribution estimation...')
    p_kde = estimate_distribution(offline_states)
    print('Online state distribution estimation...')
    q_kde = estimate_distribution(online_states)

    print('Resampling...')
    samples = p_kde.resample(cfg.batch_size)

    p = p_kde(samples)
    q = q_kde(samples)

    p = torch.tensor(p, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)

    # Avoid division by zero
    q = torch.clamp(q, min=1e-10)

    print('Computing KL divergence...')
    kl_div = F.kl_div(p.log(), q, reduction='batchmean')

    return kl_div

def init_replay_buffer(cfg, work_dir, env, offline):    
    # Compose source directory
    replay_buffer_dir = cfg.replay_buffer_dir_offline if offline else cfg.replay_buffer_dir_online
    datasets_dir = work_dir / replay_buffer_dir
    sub_str = "-td3-" if offline else "-"
    replay_dir = datasets_dir.resolve() / Path(cfg.task + sub_str + str(cfg.data_type)) / 'data'
    print(f'replay dir: {replay_dir}')

    # Construct the replay buffer
    replay_loader = make_replay_loader(env, [replay_dir], cfg.replay_buffer_size,
                                       cfg.batch_size, cfg.replay_buffer_num_workers, cfg.discount,
                                       main_task=cfg.task, task_list=[cfg.task])
    replay_iter = iter(replay_loader)

    return replay_iter

@hydra.main(config_path='.', config_name='analyse_distribution')
def main(cfg):
    # Init working directory
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    # Load data
    env = dmc.make(cfg.task, seed=cfg.seed)
    offline_replay_iter = init_replay_buffer(cfg, work_dir, env, offline=True)
    online_replay_iter = init_replay_buffer(cfg, work_dir, env, offline=False)
    print('Data loaded')

    # Measure distribution shift
    kl_div = kl_divergence(cfg, offline_replay_iter, online_replay_iter)
    print(f'KL divergence for {cfg.task}-{cfg.data_type} between teacher and offline data: {kl_div}')

if __name__ == '__main__':
    main()
