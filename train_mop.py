import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import hydra
from pathlib import Path
import dmc
import random
import torch
from replay_buffer import make_replay_loader
import utils
import wandb
import os
from logger import Logger
from video import VideoRecorder
from agent.mop import MOP
import numpy as np

NOISE_SCALAR = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 6.0, 40.0, 70.0, 65.0, 95.0, 65.0, 65.0, 80.0], dtype=np.float32)

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


def init_teacher_agent(cfg, work_dir, env, task_id, cross_teacher=False):
    '''
    Method to initialize teacher agent

    @param cfg: configuration object
    @param env: environment
    @param task_id: task of interest

    @return agent: teacher agent
    '''
    # Create agent
    agent = hydra.utils.instantiate(cfg.agent,
                                obs_shape=env.observation_spec().shape, 
                                action_shape=env.action_spec().shape,
                                num_expl_steps=0,
                                deterministic_actor=cfg.deterministic_actor)
    
    # Load weights
    teacher_dir = work_dir / cfg.teacher_dir
    task_folder = cfg.tasks[task_id]
    if not cfg.deterministic_actor:
        task_folder += '_ND'
    if cfg.data_sharing:
        task_folder += '_SHARE'
    if cross_teacher:
        task_folder += '_CROSS'
    teacher_dir = teacher_dir.resolve() / Path(task_folder) / Path(cfg.data_type[task_id])
    
    agent.load(teacher_dir)

    return agent


def eval(global_step, agent, task_id, env, logger, num_eval_episodes, video_recorder, cfg, noise_var=None):
    '''
    Method to evaluate agent in ONE environment

    @param global_step: current global step
    @param agent: agent to evaluate
    @param env: environment
    @param logger: logger
    @param num_eval_episodes: number of episodes to evaluate
    @param video_recorder: video recorder
    '''
    # Initialize variables
    step, episode, total_reward = 0, 0, 0
    if cfg.kendall:
        kt_sums = [0 for _ in range(cfg.hidden_layers+1)]
    eval_until_episode = utils.Until(num_eval_episodes)
    final_eval = (global_step == cfg.num_grad_steps - 1)
    task_name = cfg.tasks[task_id]

    # Loop over episodes
    while eval_until_episode(episode):
        # Reset environment and video recorder
        time_step = env.reset()
        video_recorder.init(env, enabled=(final_eval and episode == 0))

        # Run episode online
        while not time_step.last():
            # WARNING: eval_mode is not defined in MOP
            with torch.no_grad():
                # Retrieve state
                state = time_step.observation
                # Compute Gaussian noise
                if noise_var is not None:
                    noise = np.random.randn(*state.shape).astype(np.float32) * noise_var * NOISE_SCALAR
                else:
                    noise = np.zeros(state.shape, dtype=np.float32)
                # Select action
                if cfg.kendall:
                    action, activations = agent.act(state + noise, task_id, kendall=True)
                    for idx, activation in enumerate(activations):
                        kt_sums[idx] += activation
                else:
                    action = agent.act(state + noise, task_id)
            # Execute action
            time_step = env.step(action)
            video_recorder.record(env)
            # Sum rewards
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{task_name}_{global_step}.mp4')

    # Log results
    metrics = dict()
    reward = total_reward / episode
    length = step / episode

    if noise_var is None:
        metrics[f'episode_reward_{task_name}'] = reward
        metrics[f'episode_length_{task_name}'] = length
    else:
        metrics[f'episode_reward_{task_name}_noise_var={noise_var}'] = reward
        metrics[f'episode_length_{task_name}_noise_var={noise_var}'] = length

    if cfg.kendall:
        for idx in range(cfg.hidden_layers+1):
            metrics[f'kendall_layer{idx}_{task_name}'] = kt_sums[idx] / step

    if cfg.wandb:
        wandb.log(metrics, step=global_step)

    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', reward)
        log('episode_length', length)
        log('step', global_step)
        log('noise_variance', 0.0 if noise_var is None else noise_var)
        if cfg.kendall:
            for idx in range(cfg.hidden_layers+1):
                log(f'kendall_layer{idx}', kt_sums[idx] / step)


@hydra.main(config_path='.', config_name='config_mop')
def main(cfg):
    '''
    Main method to train multi-task offline policy

    @param cfg: configuration object
    '''
    # Init working directory
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    # Random seeds
    cfg.seed = random.randint(0, 100000)
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Initialize environments, teachers, replay buffers, and loggers for all tasks
    num_tasks = len(cfg.tasks)
    envs = []
    teachers = []
    replay_iters = []
    loggers = []
    for task_id in range(num_tasks):
        env = dmc.make(cfg.tasks[task_id], seed=cfg.seed)
        teacher = init_teacher_agent(cfg, work_dir, env, task_id)
        cross_teacher = init_teacher_agent(cfg, work_dir, env, task_id, True) if cfg.cross_teacher else None
        replay_iter = init_replay_buffer(cfg, task_id, work_dir, env, share=cfg.share_data)
        
        log_dir = work_dir / Path(f'log_{cfg.tasks[task_id]}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = Logger(log_dir, use_tb=cfg.use_tb)

        envs.append(env)
        teachers.append([teacher, cross_teacher])
        replay_iters.append(replay_iter)
        loggers.append(logger)


    # Initialize student agent
    state_dim = envs[0].observation_spec().shape[0]
    action_dim = envs[0].action_spec().shape[0]
    student = MOP(state_dim, 
                  action_dim, 
                  cfg.hidden_dim,
                  cfg.hidden_layers,
                  device, 
                  cfg.agent.lr, 
                  cfg.agent.ensemble,
                  ndcg=cfg.ndcg,
                  ndcg_alpha=cfg.ndcg_alpha,
                  ndcg_lambda=cfg.ndcg_lambda)

    # Create video recorder
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    # Set timer
    timer = utils.Timer()
    global_step = 0

    # Set training and evaluation conditions
    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    if cfg.wandb:
        wandb_dir = os.path.abspath(f"./wandb/{cfg.run_name}_{cfg.seed}")
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        wandb.init(project="mop_pd", config=cfg, name=f'{cfg.run_name}', dir=wandb_dir)
        wandb.config.update(vars(cfg))

    # Training loop
    while train_until_step(global_step):
        # Loop over tasks
        for idx in range(num_tasks):
            # Train student
            actor_loss, mse, ndcg = student.update(idx, teachers[idx][0], teachers[idx][1], replay_iters[idx], mode=cfg.mode)

            # Log metrics
            metrics = dict()
            metrics[f'actor_loss_{cfg.tasks[idx]}'] = actor_loss
            metrics[f'mse_{cfg.tasks[idx]}'] = mse
            metrics[f'ndcg_{cfg.tasks[idx]}'] = ndcg
            
            if cfg.wandb:
                wandb.log(metrics, step=global_step)
            
            loggers[idx].log_metrics(metrics, global_step, ty='train')
            if log_every_step(global_step):
                elapsed_time, total_time = timer.reset()
                with loggers[idx].log_and_dump_ctx(global_step, ty='train') as log:
                    log('fps', cfg.log_every_steps / elapsed_time)
                    log('total_time', total_time)
                    log('step', global_step)

        # Evaluate student policy on all tasks
        if eval_every_step(global_step):
            for idx in range(num_tasks):
                loggers[idx].log('eval_total_time', timer.total_time(), global_step)
                eval(global_step, student, idx, envs[idx], loggers[idx], cfg.num_eval_episodes, video_recorder, cfg, noise_var=None)
                for noise_var in cfg.noise_vars:
                    eval(global_step, student, idx, envs[idx], loggers[idx], cfg.num_eval_episodes, video_recorder, cfg, noise_var)

        # Increment global step        
        global_step += 1
    
    # Save student agent
    student.save(work_dir)


if __name__ == '__main__':
    main()