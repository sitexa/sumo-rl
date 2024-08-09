import os
import sys
from typing import Dict, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append('..')

from aits.RealWorldEnv import RealWorldEnv


class FlattenObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def _flatten_obs(self, obs):
        return np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])


class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'rewards' in info:
            self.rewards.append(info['rewards'])
        return True

    def on_training_end(self) -> None:
        # 这里你可以保存或打印累积的奖励信息
        print("Accumulated rewards:", self.rewards)


class TrainingManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm = config.get('algorithm', 'DQN')
        self.action_space_type = self._determine_action_space_type()
        self.env = self._create_env()
        self.model = None
        self.model_path = config.get('model_path', 'models')
        os.makedirs(self.model_path, exist_ok=True)

    def _determine_action_space_type(self):
        if self.algorithm in ['SAC']:
            return 'continuous'
        elif self.algorithm in ['DQN', 'PPO', 'A2C']:
            return 'discrete'
        else:
            return 'auto'

    def _create_env(self) -> gym.Env:
        env_params = self.config['env_params']
        env_params['action_space_type'] = self.action_space_type
        env = RealWorldEnv(**self.config['env_params'])
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def _create_model(self):
        if self.algorithm == 'DQN':
            if not isinstance(self.env.action_space, spaces.Discrete):
                raise ValueError("DQN requires a discrete action space.")
            self.model = DQN('MlpPolicy', self.env, **self.config.get('algo_params', {}))
        elif self.algorithm == 'PPO':
            self.model = PPO('MlpPolicy', self.env, **self.config.get('algo_params', {}))
        elif self.algorithm == 'A2C':
            self.model = A2C('MlpPolicy', self.env, **self.config.get('algo_params', {}))
        elif self.algorithm == 'SAC':
            if not isinstance(self.env.action_space, spaces.Box):
                raise ValueError("SAC requires a continuous action space.")
            self.model = SAC('MlpPolicy', self.env, **self.config.get('algo_params', {}))
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        print(f"Action space type: {self.action_space_type}")
        print(f"Action space: {self.env.action_space}")

    def train(self):
        print("=====TrainingManager-train-Starting training")
        if self.model is None:
            self._create_model()

        # Create an evaluation callback
        eval_env = self._create_env()
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.model_path,
                                     log_path=self.model_path, eval_freq=10000,
                                     deterministic=True, render=False)

        # Train the model
        reward_logger = RewardLogger()
        self.model.learn(total_timesteps=self.config.get('total_timesteps', 1000000),
                         callback=[eval_callback, reward_logger])

        # Save the final model
        self.model.save(os.path.join(self.model_path, f"final_{self.algorithm}_model"))

    def evaluate(self, num_episodes: int = 10):
        if self.model is None:
            self.load_model()

        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=num_episodes)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def save_model(self, filename: str = None):
        if filename is None:
            filename = f"{self.algorithm}_model"
        self.model.save(os.path.join(self.model_path, filename))

    def load_model(self, filename: str = None):
        if filename is None:
            filename = f"{self.algorithm}_model"
        model_cls = {'DQN': DQN, 'PPO': PPO, 'A2C': A2C, 'SAC': SAC}[self.algorithm]
        self.model = model_cls.load(os.path.join(self.model_path, filename), env=self.env)

    def visualize_training(self):
        # This method can be expanded to create more detailed visualizations
        logdir = os.path.join(self.model_path, "logs")
        os.system(f"tensorboard --logdir {logdir}")


if __name__ == "__main__":
    config = {
        'env_params': {
            'intersection_ids': ["intersection_1", "intersection_2"],
            'delta_time': 5,
            'yellow_time': 2,
            'min_green': 5,
            'max_green': 30,
            'num_seconds': 360,  # 3600
            'reward_fn': "queue"
        },
        'algorithm': 'SAC',
        'total_timesteps': 5000,  # 500000
        'algo_params': {
            'learning_rate': 1e-4,
            'buffer_size': 1000,  # 100000
            'learning_starts': 1000,
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.005,
        }
    }

    trainer = TrainingManager(config)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
