import argparse
from datetime import datetime
from time import strftime
import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from env import QEnv
from model import QModel


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1e8)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--name", type=str, default="qagent")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--full_info", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    ray.init()

    temp_env = QEnv()  
    obs_space = temp_env.observation_space

    tune.register_env("QEnv", lambda env_ctx: QEnv(**env_ctx))
    ModelCatalog.register_custom_model("QModel", QModel)
    
    now = datetime.now()
    tune.run(
        "PPO",
        config={
            "env": "QEnv",
            "env_config": {
                "full_info": args.full_info
            },
            "num_workers": args.num_workers,
            "num_gpus": 1 if args.gpu else 0,
            "framework": "torch",
            "model": {
                "custom_model": "QModel",
                "custom_model_config": {
                    "full_obs_space": obs_space
                },
            },
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True,
        checkpoint_freq=100,
        name= "qagent_" + args.name + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")
    )
