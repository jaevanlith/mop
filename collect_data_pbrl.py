import warnings
import wandb
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer_collect import ReplayBufferStorage, make_replay_loader
from video import VideoRecorder
from collections import OrderedDict

torch.backends.cudnn.benchmark = True


def init_agent(env, cfg, dir, agent_name='pbrl'):
    if agent_name == 'pbrl':
        cfg.obs_shape = env.observation_spec().shape
        cfg.action_shape = env.action_spec().shape
        cfg.num_expl_steps = 0
        cfg.deterministic_actor = True
        agent = hydra.utils.instantiate(cfg)

        agent.load(Path(dir))
    elif agent_name == 'td3':
        filename = Path(dir) / "td3.pkl"
        try:
            with open(filename, "rb") as f:
                agent = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading agent: {e}")
    else:
        raise Exception(f"Unknown agent: {cfg.agent}")
    
    return agent


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb)
        self.train_env = dmc.make(cfg.task, seed=cfg.seed)

        # create agent
        self.agent = init_agent(self.train_env, cfg.agent, cfg.agent_dir)

        # get meta specs
        meta_specs = tuple()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(), self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  replay_dir=self.work_dir / 'buffer', dataset_dir=self.work_dir / 'data')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage, cfg.replay_buffer_size,
                                                cfg.batch_size, cfg.replay_buffer_num_workers, False, 1, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def collect_data(self):
        collect_until_episode = utils.Until(self.cfg.num_collect_episodes)
        step, episode = 0, 0

        while collect_until_episode(episode):
            # reset env
            total_reward = 0
            time_step = self.train_env.reset()
            meta = OrderedDict()
            self.replay_storage.add(time_step, meta, physics=self.train_env.physics.get_state())
            self.video_recorder.init(self.train_env, enabled=(episode % 100 == 0))
            
            while not time_step.last():
                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, step=self.global_step, eval_mode=True)

                if self.cfg.noise:
                    if episode/self.cfg.num_collect_episodes < 0.1:
                        noise_var = 0
                    elif episode/self.cfg.num_collect_episodes < 0.40:
                        noise_var = 0.5
                    else:
                        noise_var = 1
                    action += np.random.randn(*action.shape).astype(np.float32) * noise_var
                elif self.cfg.random_percentage is not None:
                    if episode/self.cfg.num_collect_episodes < self.cfg.random_percentage:
                        action = np.random.uniform(-1, 1, size=action.shape).astype(np.float32)

                # take env step
                time_step = self.train_env.step(action)
                self.video_recorder.record(self.train_env)
                total_reward += time_step.reward
                step += 1
                self.replay_storage.add(time_step, meta, physics=self.train_env.physics.get_state())

                self._global_step += 1
            
            episode += 1
            self.video_recorder.save(f'{episode}.mp4')

            # Log
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_reward', total_reward)
                log('episode_length', step / episode)
                log('episode', episode)
                log('step', step)
            if self.cfg.use_wandb:
                wandb.log({"eval_return": total_reward})


@hydra.main(config_path='.', config_name='collect_data_pbrl')
def main(cfg):
    workspace = Workspace(cfg)

    if cfg.use_wandb:
        wandb_dir = f"./wandb/collect_{cfg.task}_{cfg.data_type}"
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        wandb.init(project="collect_data_pbrl", config=cfg, name=f'{cfg.task}_{cfg.data_type}', dir=wandb_dir)
        wandb.config.update(vars(cfg))

    workspace.collect_data()


if __name__ == '__main__':
    main()
