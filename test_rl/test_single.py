import gymnasium as gym
import sumo_rl

env = gym.make('sumo-rl-v0',
               net_file='nets/single-intersection/single-intersection.net.xml',
               route_file='nets/single-intersection/single-intersection.rou.xml',
               out_csv_name='test_rl/test_single.csv',
               use_gui=True,
               num_seconds=1000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
