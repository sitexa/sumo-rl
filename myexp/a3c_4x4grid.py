import os
import sys
import numpy as np
import pandas as pd
import ray
import traci
from gym import spaces
from ray.rllib.agents.a3c.a3c import A3CTrainer
# from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import sumo_rl

# Add SUMO tools to sys.path if SUMO_HOME environment variable is set
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


# Register a PettingZoo environment for a 4x4 grid simulation
register_env(
    "4x4grid",
    lambda _: PettingZooEnv(
        sumo_rl.env(
            net_file="nets/4x4-Lucas/4x4.net.xml",
            route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
            out_csv_name="outputs/4x4grid/a3c",
            use_gui=False,
            num_seconds=80000,
        )
    ),
)

config = A3CConfig()
config = config.environment("4x4grid")

config = config.training(lr=0.001, grad_clip=30.0)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=4)
print(config.to_dict())
algo = config.build()
algo.train()


# Create an A3C trainer with a policy for traffic lights
trainer = A3CTrainer(
    env="4x4grid",
    config={
        "multiagent": {
            "policies": {"0": (A3CTFPolicy, spaces.Box(low=np.zeros(11), high=np.ones(11)), spaces.Discrete(2), {})},
            "policy_mapping_fn": (lambda id: "0"),  # Traffic lights are always controlled by this policy
        },
        "lr": 0.001,
        "no_done_at_end": True,
    },
)

# Train the A3C agent
while True:
    print(trainer.train())
