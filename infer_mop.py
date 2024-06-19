from pathlib import Path
import torch
import hydra
import dmc
from agent.mop import MOP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau


def sort_and_reposition(dict_list):
    if len(dict_list) == 2:
        for i in range(3):
            dict_list[0][f'l{i+1}'], dict_list[1][f'l{i+1}'] = zip(*sorted(zip(dict_list[0][f'l{i+1}'], dict_list[1][f'l{i+1}'])))


def infer_analysis(agent, task_id, env, cfg, hidden_dim, action_dim):
    # Initialize variables
    step, l1_sum, l2_sum, l3_sum = 0, np.zeros(hidden_dim), np.zeros(hidden_dim), np.zeros(action_dim)

    # Loop over episodes
    for i in range(cfg.num_inference_episodes):
        print('Episode:', i)
        # Reset environment and video recorder
        time_step = env.reset()

        # Run episode online
        while not time_step.last():
            # WARNING: eval_mode is not defined in MOP
            with torch.no_grad():
                # Retrieve action
                a, l1_out, l2_out, l3_out = agent.infer_analysis(time_step.observation, task_id)
            # Execute action
            time_step = env.step(a)
            # Sum layer activations
            l1_sum += l1_out
            l2_sum += l2_out
            l3_sum += l3_out
            step += 1

    l1_avg = l1_sum / step
    l2_avg = l2_sum / step
    l3_avg = l3_sum / step

    return l1_avg, l2_avg, l3_avg


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

    # Initialize environments
    num_tasks = len(cfg.tasks)
    envs = []
    for task_id in range(num_tasks):
        env = dmc.make(cfg.tasks[task_id], seed=cfg.seed)
        envs.append(env)

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
    for idx in range(num_tasks):
        print(f'Inferencing task {cfg.tasks[idx]}')
        l1_avg, l2_avg, l3_avg = infer_analysis(student, idx, envs[idx], cfg, hidden_dim, action_dim)

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

    # Calculate Kendall's tau
    tau_l1 = kendalltau(dict_list[0]['l1'], dict_list[1]['l1'])[0]
    tau_l2 = kendalltau(dict_list[0]['l2'], dict_list[1]['l2'])[0]
    tau_l3 = kendalltau(dict_list[0]['l3'], dict_list[1]['l3'])[0]
    print(f'Kendall\'s tau for layer 1: {tau_l1}')
    print(f'Kendall\'s tau for layer 2: {tau_l2}')
    print(f'Kendall\'s tau for layer 3: {tau_l3}')
    tau_df = pd.DataFrame({'Layer 1': [tau_l1], 'Layer 2': [tau_l2], 'Layer 3': [tau_l3]})
    tau_df = tau_df.set_index('Layer 1')
    tau_df.to_csv(work_dir / Path('kendall_tau.csv'))

if __name__ == '__main__':
    main()