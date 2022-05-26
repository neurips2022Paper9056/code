# Install
Install C++ code by running
```
make
```
Install gym environment for RL with pip:
```
cd gym-door-nerf2
pip install -e .
```

# Train encodings
There are seperate files for training each of the encoding methods. For example, in order to train the NeRF-RL comp. + field encoding, run
```
python CompNeRFModel.py
```

# RL training and evaluation
## Specify Experiment ID and Run ID
Experiments are identified by their 3-digit ID, e.g.,
```
export CONFIG_ID=007
```
Independent runs are specified using, e.g.,
```
SLURM_ARRAY_TASK_ID=test
```

## Train RL
After setting `$CONFIG_ID` and `$SLURM_ARRAY_TASK_ID`, run
```
python 001_training.py
```

## Evaluate RL
After setting `$CONFIG_ID` and `$SLURM_ARRAY_TASK_ID`, run
```
python 002_evaluation.py
```

## Visualize agent
Edit `CONFIG_ID` (and potentially `AGENT_ID` and `TIMESTEP`) in `agent_playback.py`, then run
```
python agent_playback.py
```
to view an interactive visualization of the RL agent.