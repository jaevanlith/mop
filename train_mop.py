import hydra
from pathlib import Path
import dmc
import random
import torch
from replay_buffer import make_replay_loader
import utils
from logger import Logger
from video import VideoRecorder
from agent.mop import MOP

def init_replay_buffer(cfg, task_id, work_dir, env):
    '''
    Method to initialize replay buffer for a given task

    @param cfg: configuration object
    @param task_id: task of interest
    @param work_dir: working directory

    @return replay_iter: replay buffer iterator
    '''
    # Compose source directory
    task = cfg.tasks[task_id]
    data_type = cfg.data_type[task_id]
    datasets_dir = work_dir / cfg.replay_buffer_dir
    replay_dir = datasets_dir.resolve() / Path(task+"-td3-"+str(data_type)) / 'data'
    print(f'replay dir: {replay_dir}')

    # Construct the replay buffer
    replay_loader = make_replay_loader(env, [replay_dir], cfg.replay_buffer_size,
				cfg.batch_size, cfg.replay_buffer_num_workers, cfg.discount,
				main_task=task, task_list=[task])
    replay_iter = iter(replay_loader)

    return replay_iter


def init_teacher_agent(cfg, work_dir, env, task_id):
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
                                num_expl_steps=0)
    
    # Load weights
    teachers_dir = work_dir / cfg.teacher_dir
    agent_dir = teachers_dir.resolve() / Path(cfg.tasks[task_id])
    agent.load(agent_dir)

    return agent


def eval(global_step, agent, task_id, env, logger, num_eval_episodes, video_recorder):
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
    eval_until_episode = utils.Until(num_eval_episodes)

    # Loop over episodes
    while eval_until_episode(episode):
        # Reset environment and video recorder
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))

        # Run episode online
        while not time_step.last():
            # WARNING: eval_mode is not defined in MOP
            with torch.no_grad():
                # Retrieve action
                action = agent.act(time_step.observation, task_id)
            # Execute action
            time_step = env.step(action)
            video_recorder.record(env)
            # Sum rewards
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{global_step}.mp4')

    # Log results
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', total_reward / episode)
        log('episode_length', step / episode)
        log('step', global_step)


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

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    # Initialize environments, teachers, and replay buffers for all tasks
    num_tasks = len(cfg.tasks)
    envs = []
    teachers = []
    replay_iters = []
    for task_id in range(num_tasks):
        env = dmc.make(cfg.tasks[task_id], seed=cfg.seed)
        teacher = init_teacher_agent(cfg, work_dir, env, task_id)
        replay_iter = init_replay_buffer(cfg, task_id, work_dir, env)

        envs.append(env)
        teachers.append(teacher)
        replay_iters.append(replay_iter)

    # Initialize student agent
    state_dim = envs[0].observation_spec().shape[0]
    action_dim = envs[0].action_spec().shape[0]
    lr = cfg.agent.lr
    student = MOP(state_dim=state_dim, action_dim=action_dim, device=device, lr=lr)

    # Create video recorder
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    # Set timer
    timer = utils.Timer()
    global_step = 0

    # Set training and evaluation conditions
    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    # Training loop
    while train_until_step(global_step):
        
        # Evaluate student policy on all tasks
        if eval_every_step(global_step):
            logger.log('eval_total_time', timer.total_time(), global_step)
            # TODO: evaluate on ALL environments
            task_id = 0
            eval(global_step, student, task_id, envs[0], logger, cfg.num_eval_episodes, video_recorder)

        # Loop over tasks
        for idx in range(num_tasks):
            # Train student
            metrics = student.update_actor(idx, teachers[idx], replay_iters[idx]) 
            
            # Log metrics
            logger.log_metrics(metrics, global_step, ty='train')
            if log_every_step(global_step):
                elapsed_time, total_time = timer.reset()
                with logger.log_and_dump_ctx(global_step, ty='train') as log:
                    log('fps', cfg.log_every_steps / elapsed_time)
                    log('total_time', total_time)
                    log('step', global_step)

        # Increment global step        
        global_step += 1



if __name__ == '__main__':
    main()