import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import random
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
import json
from pathlib import Path
import hydra
import numpy as np
import torch
from dm_env import specs
import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
import copy

torch.backends.cudnn.benchmark = True

with open("task.json", "r") as f:
	task_dict = json.load(f)


def get_domain(task):
	if task.startswith('point_mass_maze'):
		return 'point_mass_maze'
	return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
	return (seed - 1) % num_data_seeds + 1


def eval(global_step, agent, task_id, env, logger, num_eval_episodes, video_recorder, cfg):
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
                action = agent.act(time_step.observation, step=global_step, eval_mode=True)
            # Execute action
            time_step = env.step(action)
            video_recorder.record(env)
            # Sum rewards
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{global_step}.mp4')

    # Log results
    metrics = dict()
    task_name = cfg.tasks[task_id]
    metrics[f'episode_reward_{task_name}'] = total_reward / episode
    metrics[f'episode_length_{task_name}'] = step / episode

    if cfg.wandb:
        wandb.log(metrics, step=global_step)

    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', total_reward / episode)
        log('episode_length', step / episode)
        log('step', global_step)


def init_replay_buffer(cfg, task_id, work_dir, env):
	'''
	Method to initialize replay buffer

	@param cfg: configuration object
	@param task_id: task of interest
	@param work_dir: working directory
	@param env: environment

	@return replay_iter: replay buffer iterator
	'''
	replay_dir_list = []

	for idx in range(len(cfg.share_task)):
		task = cfg.share_task[idx]          # dataset task
		data_type = cfg.data_type[idx]      # dataset type [random, medium, medium-replay, expert, replay]
		datasets_dir = work_dir / cfg.replay_buffer_dir
		replay_dir = datasets_dir.resolve() / Path(task+"-td3-"+str(data_type)) / 'data'
		print(f'replay dir: {replay_dir}')
		replay_dir_list.append(replay_dir)

	# construct the replay buffer. env is the main task, we use it to relabel the reward of other tasks
	replay_loader = make_replay_loader(env, replay_dir_list, cfg.replay_buffer_size,
				cfg.batch_size, cfg.replay_buffer_num_workers, cfg.discount,
				main_task=cfg.tasks[task_id], task_list=cfg.share_task)
	print("load data...")
	replay_iter = iter(replay_loader)     # OfflineReplayBuffer sample function
	print("load data done.")

	return replay_iter


def init_agent(cfg, env, task_id):
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
								task_id=task_id,
                                num_expl_steps=0)

    return agent


@hydra.main(config_path='.', config_name='config_pbrl_mt')
def main(cfg):
	work_dir = Path.cwd()
	print(f'workspace: {work_dir}')

	# random seeds
	cfg.seed = random.randint(0, 100000)
	utils.set_seed_everywhere(cfg.seed)
	device = torch.device(cfg.device)

	# Initialize environments, teachers, replay buffers, and loggers for all tasks
	num_tasks = len(cfg.tasks)
	envs = []
	agents = []
	replay_iters = []
	loggers = []
	for task_id in range(num_tasks):
		env = dmc.make(cfg.tasks[task_id], seed=cfg.seed)
		agent = init_agent(cfg, env, task_id)
		replay_iter = init_replay_buffer(cfg, task_id, work_dir, env)
		
		log_dir = work_dir / Path(f'log_{cfg.tasks[task_id]}')
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		logger = Logger(log_dir, use_tb=cfg.use_tb)

		envs.append(env)
		agents.append(agent)
		replay_iters.append(replay_iter)
		loggers.append(logger)

	# create video recorders
	video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

	timer = utils.Timer()
	global_step = 0

	train_until_step = utils.Until(cfg.num_grad_steps)
	eval_every_step = utils.Every(cfg.eval_every_steps)
	log_every_step = utils.Every(cfg.log_every_steps)

	if cfg.wandb:
		path_str = f'{cfg.agent.name}_{cfg.share_task[0]}_{cfg.share_task[1]}_{cfg.data_type[0]}_{cfg.data_type[1]}'
		wandb_dir = f"./wandb/{path_str}_{cfg.seed}"
		if not os.path.exists(wandb_dir):
			os.makedirs(wandb_dir)
		wandb.init(project="mop_pd", config=cfg, name=f'{path_str}_1', dir=wandb_dir)
		wandb.config.update(vars(cfg))

	while train_until_step(global_step):

		# Evaluate student policy on all tasks
		if eval_every_step(global_step):
			for idx in range(num_tasks):
				loggers[idx].log('eval_total_time', timer.total_time(), global_step)
				eval(global_step, agents[idx], idx, envs[idx], loggers[idx], cfg.num_eval_episodes, video_recorder, cfg)

		# Loop over tasks
		for idx in range(num_tasks):
			# train the agent
			output_dict = agents[idx].update(replay_iters[idx], global_step, cfg.num_grad_steps)

			# Copy actor to all other agents
			for i in range(num_tasks):
				if i != idx:
					agents[i].actor = copy.deepcopy(agents[idx].actor)

			# log metrics
			if 'actor_loss' in output_dict:
				actor_loss = output_dict['actor_loss']
				metrics = dict()
				metrics[f'actor_loss_{cfg.tasks[idx]}'] = actor_loss
				if cfg.wandb:
					wandb.log(metrics, step=global_step)

				# log
				loggers[idx].log_metrics(metrics, global_step, ty='train')
				if log_every_step(global_step):
					elapsed_time, total_time = timer.reset()
					with loggers[idx].log_and_dump_ctx(global_step, ty='train') as log:
						log('fps', cfg.log_every_steps / elapsed_time)
						log('total_time', total_time)
						log('step', global_step)

		global_step += 1
		
	# Save final agents
	for idx in range(num_tasks):
		agents[idx].save(work_dir, Path(f'models_{cfg.tasks[idx]}'))


if __name__ == '__main__':
	main()
