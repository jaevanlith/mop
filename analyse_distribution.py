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

def kl_divergence(cfg, offline_replay_iter, online_replay_iter, epsilon=1e-10):
    offline_states = next(offline_replay_iter)[0]
    online_states = next(online_replay_iter)[0]
    
    assert offline_states.shape == online_states.shape

    print('Retrieveing min and max state values...')
    min_val = torch.minimum(offline_states.min(dim=0)[0], online_states.min(dim=0)[0])
    max_val = torch.maximum(offline_states.max(dim=0)[0], online_states.max(dim=0)[0])
    min_max_vals = tuple(x.item() for x in sum(zip(min_val, max_val), ()))

    p_hist = torch.histogramdd(offline_states, bins=cfg.bins, range=min_max_vals)[0]
    q_hist = torch.histogramdd(online_states, bins=cfg.bins, range=min_max_vals)[0]

    p = p_hist + epsilon
    q = q_hist + epsilon
    
    p = p / p.sum()
    q = q / q.sum()

    # Avoid division by zero
    q = torch.clamp(q, min=1e-10)

    print('Computing KL divergence...')
    kl_div = F.kl_div(p.log(), q, reduction='batchmean')

    return kl_div

def init_replay_buffer(cfg, task, data_type, work_dir, env, offline):    
    # Compose source directory
    replay_buffer_dir = cfg.replay_buffer_dir_offline if offline else cfg.replay_buffer_dir_online
    datasets_dir = work_dir / replay_buffer_dir
    sub_str = "-td3-" if offline else "-"
    replay_dir = datasets_dir.resolve() / Path(task + sub_str + str(data_type)) / 'data'
    print(f'replay dir: {replay_dir}')

    # Construct the replay buffer
    replay_loader = make_replay_loader(env, [replay_dir], cfg.replay_buffer_size,
                                       cfg.batch_size, cfg.replay_buffer_num_workers, cfg.discount,
                                       main_task=task, task_list=[task])
    replay_iter = iter(replay_loader)

    return replay_iter

@hydra.main(config_path='.', config_name='analyse_distribution')
def main(cfg):
    # Init working directory
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    # Load data
    if cfg.mode == "BxT":
        env = dmc.make(cfg.task[0], seed=cfg.seed)
        replay_iter1 = init_replay_buffer(cfg, cfg.task[0], cfg.data_type[0], work_dir, env, offline=True)
        replay_iter2 = init_replay_buffer(cfg, cfg.task[0], cfg.data_type[0], work_dir, env, offline=False)
    elif cfg.mode == "TxT":
        env1 = dmc.make(cfg.task[0], seed=cfg.seed)
        env2 = dmc.make(cfg.task[1], seed=cfg.seed)
        replay_iter1 = init_replay_buffer(cfg, cfg.task[0], cfg.data_type[0], work_dir, env1, offline=False)
        replay_iter2 = init_replay_buffer(cfg, cfg.task[1], cfg.data_type[1], work_dir, env2, offline=False)
    else:
        raise ValueError(f'Invalid mode: {cfg.mode}')
    print('Data loaded')

    # Measure distribution shift
    kl_div = kl_divergence(cfg, replay_iter1, replay_iter2)
    if cfg.mode == "BxT":
        result = f'KL divergence between behavior and teacher policy for {cfg.task[0]}-{cfg.data_type[0]}: {kl_div}'
    elif cfg.mode == "TxT":
        result = f'KL divergence between teacher policy {cfg.task[0]}-{cfg.data_type[0]} and {cfg.task[1]}-{cfg.data_type[1]}: {kl_div}'
    print(result)

    f = open("./result.txt", "w")
    f.write(result)
    f.close()

if __name__ == '__main__':
    main()
