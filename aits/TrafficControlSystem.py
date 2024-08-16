import json
import sys
import time
import os
import numpy as np
from typing import Dict, Any

sys.path.append('..')

from aits.RealWorldEnv import RealWorldEnv
from stable_baselines3 import PPO, SAC, A2C, DQN


class TrafficControlSystem:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.env = RealWorldEnv(**self.config.get("env_params"))
        self.algorithm = self.config.get("algorithm", "DQN")
        self.model_path = self.config.get("model_path", "")
        self.model = None
        self.load_model()

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as f:
            return json.load(f)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.algorithm == 'SAC':
            self.model = SAC.load(self.model_path)
        elif self.algorithm == 'PPO':
            self.model = PPO.load(self.model_path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(self.model_path)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(self.model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def get_action(self, observation):
        # 处理整个环境的观察
        if isinstance(observation, dict):
            # 如果观察是字典（每个交叉口一个观察），我们需要将它们连接起来
            obs_array = np.concatenate([obs for obs in observation.values()])
        else:
            # 如果观察已经是一个数组，直接使用它
            obs_array = observation

        # 确保观察的形状正确
        expected_shape = self.model.observation_space.shape
        if obs_array.shape != expected_shape:
            # 如果形状不匹配，进行适当的调整
            if len(obs_array) < expected_shape[0]:
                # 如果观察太短，填充零
                padded_obs = np.zeros(expected_shape)
                padded_obs[:len(obs_array)] = obs_array
                obs_array = padded_obs
            else:
                # 如果观察太长，截断
                obs_array = obs_array[:expected_shape[0]]

        action, _ = self.model.predict(obs_array, deterministic=True)
        return action

    def run(self):
        obs, _ = self.env.reset()
        try:
            while True:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                print(f"Step info: {info}")
                time.sleep(self.env.delta_time)

                if terminated or truncated:
                    obs, _ = self.env.reset()
        except KeyboardInterrupt:
            print("Stopping the system...")
        finally:
            self.env.close()


def main():
    config_file = "config/ppo-run.json"
    tcs = TrafficControlSystem(config_file)
    tcs.run()


if __name__ == "__main__":
    main()
