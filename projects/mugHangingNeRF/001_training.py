# %%
"""
RL training script
"""
import os
import json
import torch
import gym
import numpy as np

from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback

# %%
# parameters
with open("configs/" + os.environ["CONFIG_ID"] + ".json", 'r', encoding="utf8") as config_data:
    config = json.load(config_data)

scratch_dir = os.path.join(
    config["scratch_root"],
    os.environ["CONFIG_ID"]
)
print(f'Scratch directory: {scratch_dir}')
if not os.path.isdir(scratch_dir):
    print(f'Creating new directory: {scratch_dir}')
    os.mkdir(scratch_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'PyTorch runs on device "{device}"')
print(f'Running config {os.environ["CONFIG_ID"]}: "{config["description"]}"')

# %%
# create env
env = gym.make(
    'gym_mug_nerf:mug-nerf-v0',
    exp_path=config["exp_path"],
    network_number=config["network_number"],
    device=device,
    encoder_model=config["encoder_model"],
    new_mug_every_n_resets=config["new_mug_every_n_resets"],
    reset_int=config["reset_int"],
    reset_position=config["reset_position"],
    observe_3d_state=config["observe_3d_state"],
    use_keypoints=config["use_keypoints"],
    use_fourier = config["use_fourier"] if "use_fourier" in config else False
)

# The environment does not have a time limit itself, but
# this can be provided using the TimeLimit wrapper
env = gym.wrappers.TimeLimit(
    env, max_episode_steps=config["max_episode_steps"])

# %%
# create agent
if config['model_class'] == 'PPO':
    ModelClass = PPO
    on_policy=True
elif config['model_class'] == 'DDPG':
    ModelClass = DDPG
    on_policy=False
elif config['model_class'] == 'SAC':
    ModelClass = SAC
    on_policy=False
elif config['model_class'] == 'TD3':
    ModelClass = TD3
    on_policy=False
else:
    raise Exception(f"Unknown model_class: {config['model_class']}")

kwargs = dict(
    verbose=1,
    device=device,
    policy_kwargs={},
    gamma=config["gamma"]
)
if not on_policy:
    kwargs["buffer_size"]=config["buffer_size"]

model = ModelClass(
    config['policy'],
    env,
    **kwargs
)

# %%
# load existing checkpoints
name_prefix = os.environ["SLURM_ARRAY_TASK_ID"].zfill(
    config["file_string_digits"]
)

if config["pickup_checkpoint"]:
    newest_path = None
    newest_checkpoint = None
    checkpoints = np.arange(
        config["total_timesteps"],
        step=config["save_interval"]
    ) + config["save_interval"]
    for checkpoint in checkpoints:
        path = os.path.join(
            scratch_dir,
            name_prefix + "_" + str(checkpoint) + "_steps"
        )
        if os.path.isfile(path + ".zip"):
            newest_path = path
            newest_checkpoint = checkpoint

    if newest_path is not None:
        # .load() is a class method and instantiates the agent
        model = ModelClass.load(newest_path, env=env, device=device)
        model.verbose = 1
        name_prefix = name_prefix + "_" + str(newest_checkpoint) + "_offset"
        print(f"Start with existing model after {newest_checkpoint} steps")
        print(f"Model is loaded from {newest_path}")
        print(f"Name prefix was modified to {name_prefix}")

# %%
callback = CheckpointCallback(
    save_freq=config["save_interval"],
    save_path=scratch_dir,
    name_prefix=name_prefix
)
model.learn(
    config["total_timesteps"],
    callback=callback
)

# %%
