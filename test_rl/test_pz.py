import sumo_rl

env = sumo_rl.env(net_file='sumo_net_file.net.xml',
                  route_file='sumo_route_file.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation)
    env.step(action)
