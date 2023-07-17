from ray import tune
from ray.rllib.algorithms.a3c import A3CConfig

config = A3CConfig()
# config = config.training(lr=0.01, grad_clip=30.0)
config = config.training(lr=tune.grid_search([0.001, 0.0001]), use_critic=False)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=4)
config = config.environment("CartPole-v1")
# print(config.to_dict())
# algo = config.build()
# algo.train()
tune.Tuner("A3C", stop={"episode_reward_mean": 200}, param_space=config.to_dict(),).fit()
