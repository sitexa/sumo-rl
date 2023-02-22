import gym

gym.register(
    id='CityTraffic-v0',
    entry_point='myenv:city_traffic.CityTrafficEnv',
    max_episode_steps=100,
    reward_threshold=1.0,
)
