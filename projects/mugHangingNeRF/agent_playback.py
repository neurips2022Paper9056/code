# %%
import os
import json
import torch
import time
import gym
import numpy as np

import plot_utils

from stable_baselines3 import PPO, DDPG, SAC, TD3

CONFIG_ID="008"
AGENT_ID=None # None means automatic selection
TIMESTEP=None

# parameters
with open("configs/" + CONFIG_ID + ".json", 'r', encoding="utf8") as config_data:
    config = json.load(config_data)

scratch_dir = os.path.join(
    config["scratch_root"],
    CONFIG_ID
)
print(f'Scratch directory: {scratch_dir}')
assert os.path.isdir(scratch_dir), f"Scratch dir {scratch_dir} does not exist"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'PyTorch runs on device "{device}"')
print(f'Running config {CONFIG_ID}: "{config["description"]}"')

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

# create agent
if config['model_class'] == 'PPO':
    ModelClass = PPO
elif config['model_class'] == 'DDPG':
    ModelClass = DDPG
elif config['model_class'] == 'SAC':
    ModelClass = SAC
elif config['model_class'] == 'TD3':
    ModelClass = TD3
else:
    raise Exception(f"Unknown model_class: {config['model_class']}")

# determine model to load
agents, timesteps, successes, final_distances, rollout_steps = plot_utils.read_in_results(
    CONFIG_ID
)
mean_success = np.mean(successes, axis=-1)

if (AGENT_ID is None) or (TIMESTEP is None):
    # in this case, automatically choose the best agent
    plot_agent_ind, plot_timestep_ind = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.nanargmax(mean_success), mean_success.shape
    )
    AGENT_ID = agents[plot_agent_ind]
    TIMESTEP = timesteps[plot_timestep_ind]
else:
    plot_agent_ind = int(np.where(agents == AGENT_ID)[0][0])
    plot_timestep_ind = int(np.where(timesteps == TIMESTEP)[0][0])

success_rate = mean_success[plot_agent_ind, plot_timestep_ind]
print(f"Success rate: {success_rate} Agent: {AGENT_ID} Timestep: {TIMESTEP}")

filename = str(AGENT_ID).zfill(
    config["file_string_digits"]
) + "_" + str(TIMESTEP) + '_steps'
model_path = os.path.join(scratch_dir, filename)


# load model
model = ModelClass.load(os.path.join(
    scratch_dir,
    filename
), env=env, device=device)

env.render()

# %%
# run simulation
obs = env.reset()

# %%
for timestep in range(config["max_episode_steps"]):
    print(timestep)
    action, _ = model.predict(
        obs, deterministic=config["eval_deterministic"])
    obs, reward, done, info = env.step(action)
    time.sleep(0.1)

# %%
