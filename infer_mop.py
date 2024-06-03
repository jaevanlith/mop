from pathlib import Path
import torch
import hydra
import os
import dmc
from replay_buffer import make_replay_loader
from logger import Logger
from agent.mop import MOP
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init_replay_buffer(cfg, task_id, work_dir, env, share=False):
    '''
    Method to initialize replay buffer for a given task

    @param cfg: configuration object
    @param task_id: task of interest
    @param work_dir: working directory
    @param env: environment
    @param share: whether to share replay buffer across tasks

    @return replay_iter: replay buffer iterator
    '''
    if share:
        task_id_list = range(len(cfg.tasks))
    else:
        task_id_list = [task_id]
    
    # Compose source directory
    replay_dir_list = []
    for idx in task_id_list:
        task = cfg.tasks[idx]
        data_type = cfg.data_type[idx]
        datasets_dir = work_dir / cfg.replay_buffer_dir
        replay_dir = datasets_dir.resolve() / Path(task+"-td3-"+str(data_type)) / 'data'
        print(f'replay dir: {replay_dir}')
        replay_dir_list.append(replay_dir)

    # Construct the replay buffer3
    replay_loader = make_replay_loader(env, replay_dir_list, cfg.replay_buffer_size,
				cfg.batch_size, cfg.replay_buffer_num_workers, cfg.discount,
				main_task=task, task_list=[cfg.tasks[idx] for idx in task_id_list])
    replay_iter = iter(replay_loader)

    return replay_iter


def sort_and_reposition(dict_list):
    if len(dict_list) == 2:
        for i in range(3):
            dict_list[0][f'l{i+1}'], dict_list[1][f'l{i+1}'] = zip(*sorted(zip(dict_list[0][f'l{i+1}'], dict_list[1][f'l{i+1}'])))
    else:
        return dict_list


def plot_results(dict_list, sorted, cfg):
    # Create the subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot each dictionary's values in the subplots
    for idx, d in enumerate(dict_list):
        axs[0].plot(d['l1'], label=f'{cfg.tasks[idx]}')
        axs[1].plot(d['l2'], label=f'{cfg.tasks[idx]}')
        axs[2].plot(d['l3'], label=f'{cfg.tasks[idx]}')

    # Set titles and legends
    for i, ax in enumerate(axs):
        ax.set_title(f'Layer {i+1}')
        ax.legend()
        if sorted:
            ax.set_xlabel(f'Neurons sorted on {cfg.tasks[0]} activation value (ascending)')
        else:
            ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Activation value')

    # Display the plot
    plt.tight_layout(h_pad=3)
    sorted_string = '_sorted' if sorted else ''
    plt.savefig(f'plot{sorted_string}.png')


@hydra.main(config_path='.', config_name='config_mop_infer')
def main(cfg):
    # Init working directory
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    device = torch.device(cfg.device)

    # Initialize replay buffers
    num_tasks = len(cfg.tasks)
    envs = []
    replay_iters = []
    for task_id in range(num_tasks):
        env = dmc.make(cfg.tasks[task_id], seed=cfg.seed)
        replay_iter = init_replay_buffer(cfg, task_id, work_dir, env, share=False)

        envs.append(env)
        replay_iters.append(replay_iter)

    # Init student agent
    state_dim = envs[0].observation_spec().shape[0]
    action_dim = envs[0].action_spec().shape[0]
    hidden_dim = cfg.hidden_dim
    lr = cfg.lr
    ensemble = cfg.ensemble
    student = MOP(state_dim, action_dim, hidden_dim, device, lr, ensemble)

    # Load student
    student_dir = work_dir / cfg.student_dir
    student_dir = student_dir.resolve() / Path(f'{cfg.tasks[0]}_{cfg.data_type[0]}_{cfg.tasks[1]}_{cfg.data_type[1]}')
    student.load(student_dir)

    dict_list = []
    for idx in [1,1]:
        l1_sum = np.zeros(hidden_dim)
        l2_sum = np.zeros(hidden_dim)
        l3_sum = np.zeros(action_dim)

        for _ in range(cfg.num_inference_steps):
            # Sample batch of data
            batch = next(replay_iters[idx])
            state, action, reward, discount, next_obs, bool_flag = utils.to_torch(batch, cfg.device)

            if cfg.share_states:
                for i in range(num_tasks):
                    if i != idx:
                        batch = next(replay_iters[i])
                        state_extra, action, reward, discount, next_obs, bool_flag = utils.to_torch(batch, cfg.device)
                        state = torch.cat((state, state_extra), dim=0)

            # Perform inference
            l1_out, l2_out, l3_out = student.infer_analysis(state, idx)

            # Update sums
            l1_sum += l1_out
            l2_sum += l2_out
            l3_sum += l3_out
        
        # Compute averages
        l1_avg = l1_sum / cfg.num_inference_steps
        l2_avg = l2_sum / cfg.num_inference_steps
        l3_avg = l3_sum / cfg.num_inference_steps

        # Save results
        dict = {'l1': l1_avg, 'l2': l2_avg, 'l3': l3_avg}
        dict_list.append(dict)
        df = pd.DataFrame.from_dict(dict, orient='index')
        df = df.transpose()
        df.index.name = 'node_id'
        df.to_csv(work_dir / Path(f'inference_{cfg.tasks[idx]}.csv'))

    # Plot results
    plot_results(dict_list, False, cfg)
    sort_and_reposition(dict_list)
    plot_results(dict_list, True, cfg)


if __name__ == '__main__':
    main()