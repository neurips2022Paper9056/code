import os
import sys
import json
import numpy as np
import gym
import torch

import CompNeRFModel
import CNNCNN
import CURL
import CURLSingleView
import CompCNNEncoder
from src.contrastiveLoss import centerCrop

sys.path.append(os.getenv("HOME") + '/git/imorl/projects/doorOpeningNeRF2')
import doorEnvironment

class DoorEnvNerf2(gym.Env):
    """
    Gym wrapper for doorEnvironment
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        exp_path,
        network_number,
        device,
        encoder_model,
        wall_offset,
        new_handle_every_n_resets=1, # None corresponds to never
        reset_int=None, # None corresponds to random choice at each reset
        reset_position=None,
        observe_3d_state=False,
        use_keypoints=False,
        use_fourier=False,
        perturb_masks=None,
        max_action=0.02
    ):
        self.device = device
        self.encoder_model = encoder_model
        self.new_handle_every_n_resets = new_handle_every_n_resets
        self.reset_int = reset_int
        self.reset_position=torch.Tensor(reset_position) if reset_position is not None else None
        self.observe_3d_state=observe_3d_state
        self.use_keypoints=use_keypoints
        self.use_fourier=use_fourier
        self.perturb_masks=perturb_masks
        self.max_action = max_action
        self.wall_offset = wall_offset

        if not self.observe_3d_state:
            self.exp_config = json.load(open(exp_path + '/config.json'))
            if self.encoder_model == "Comp3DDyn":
                self.comp3DDyn = CompNeRFModel.Comp3DDyn(**self.exp_config)
                self.comp3DDyn.buildModel()
                self.comp3DDyn.loadNetwork(exp_path + '/' + str(network_number))
            elif self.encoder_model == "CNNCNN":
                self.cnncnn = CNNCNN.CNNCNN(**self.exp_config)
                self.cnncnn.buildModel()
                self.cnncnn.loadNetwork(exp_path + '/' + str(network_number))
                self.cnncnn.objectEncoder.eval()
            elif self.encoder_model == "CNNNeRF":
                self.cnnnerf = CompCNNEncoder.CompCNNEncoderNeRF(**self.exp_config)
                self.cnnnerf.buildModel()
                self.cnnnerf.loadNetwork(exp_path + '/' + str(network_number))
                self.cnnnerf.objectEncoder.eval()
                if 'global' not in self.exp_config: self.exp_config['global'] = False
            elif self.encoder_model == "ConPre":
                self.curl = CURLSingleView.ConPre(**self.exp_config)
                self.curl.buildModel()
                self.curl.loadNetwork(exp_path + '/' + str(network_number))
                self.curl.objectEncoder.eval() # not important here, but better be safe
            elif self.encoder_model == "TimeConPre":
                self.curltime = CURL.TimeConPre(**self.exp_config)
                self.curltime.buildModel()
                self.curltime.loadNetwork(exp_path + '/' + str(network_number))
            else:
                raise Exception("Unknown encoder_model")

        if self.new_handle_every_n_resets is None:
            assert self.reset_int is not None, "Mug shape must be configured if it is not random"
        else:
            assert self.reset_int is None, "You configured random handle resets AND a handle shape"

        if self.observe_3d_state and not self.use_keypoints:
            assert self.reset_int is not None, "Need constant shape for 3D state (otherwise MDP is ill-posed)"

        if self.use_keypoints:
            assert self.observe_3d_state, "Keypoints only available for 3d state"

        if self.use_fourier:
            assert self.observe_3d_state, "fourier only available for 3d state"
            assert self.reset_int is not None, "Need constant shape for 3D state (otherwise MDP is ill-posed)"

        if self.reset_position is not None:
            assert self.reset_position[0] == 0.0
            assert self.reset_position[2] == -0.1

        if self.perturb_masks is not None:
            assert not self.observe_3d_state, "Mask perturbation has no effect on 3D state"

        self.environment = doorEnvironment.DoorEnvironment()
        self.watch = False
        self.environment.initEnvironment(
            self.reset_int if self.reset_int is not None else np.random.randint(1000000000),
            self.wall_offset
        )

        state_bounds = self.environment.getStateBounds()
        self.state_3d_low = np.array([
            state_bounds[0][0],
            state_bounds[1][0],
            state_bounds[2][0],
            state_bounds[3][0]
        ])
        self.state_3d_high = np.array([
            state_bounds[0][1],
            state_bounds[1][1],
            state_bounds[2][1],
            state_bounds[3][1]
        ])

        self.state_3d_space = gym.spaces.Box(
            low=self.state_3d_low,
            high=self.state_3d_high
        )

        # determine observation space
        if not self.observe_3d_state:
            if self.encoder_model == "Comp3DDyn":
                # Nerf encoding
                obs_shape = (2*self.comp3DDyn.C["k"],) # 2 objects
            elif self.encoder_model == "CNNCNN":
                # CNN encoding
                if self.exp_config['global']:
                    obs_shape = (self.cnncnn.C["k"],)
                else:
                    obs_shape = (2*self.cnncnn.C["k"],)
            elif self.encoder_model == "CNNNeRF":
                # CNN encoding with NeRF decoding
                if self.exp_config['global']:
                    obs_shape = (self.cnnnerf.C["k"],)
                else:
                    obs_shape = (2*self.cnnnerf.C["k"],)
            elif self.encoder_model == "ConPre":
                # CURL encoding
                obs_shape = (self.curl.C["k"],)
            elif self.encoder_model == "TimeConPre":
                # CURL time contrastive encoding
                obs_shape = (2*self.curltime.C["k"],)
            else:
                raise Exception("Unknown encoder_model")
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=obs_shape
            )

        if self.observe_3d_state:
            if self.use_keypoints:
                xlim = [-0.15-0.85/2, -0.15+0.85/2]
                ylim = [-0.03-0.2/2, -0.03+0.2/2]
                zlim = [0.2-0.4/2, 0.2+0.4/2]

                lower = np.array(5*[
                    xlim[0],
                    ylim[0],
                    zlim[0]
                ])
                upper = np.array(5*[
                    xlim[1],
                    ylim[1],
                    zlim[1]
                ])
                self.observation_space = gym.spaces.Box(
                    low=lower,
                    high=upper
                )
            elif self.use_fourier:
                obs_shape = (80,)
                self.observation_space = gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=obs_shape
                )
            else:
                self.observation_space = self.state_3d_space

        # determine action space
        self.action_space = gym.spaces.Box(
            low=-max_action*np.ones(3),
            high=max_action*np.ones(3)
        )

        self.n_resets = 0
        self.reset()

    def reset(self):
        """
        Reset
        """
        # reset mug shape
        if self.new_handle_every_n_resets is not None:
            if self.n_resets % self.new_handle_every_n_resets==0:
                self.environment.initEnvironment(
                    self.reset_int if self.reset_int is not None else np.random.randint(1000000000),
                    self.wall_offset
                )

        # and increment self.n_resets
        self.n_resets += 1

        if self.reset_position is None:
            # randomize initial position
            state = self.state_3d_space.sample()
            # but always reset door at 0.0
            state[0] = 0.0
            # and gripper at y0 = -0.1
            state[2] = -0.1
            self._controlled_reset(state)

        else:
            self._controlled_reset(self.reset_position)

        # Renew watch() if render was activated
        if self.watch:
            self.environment.watch()

        # return observation
        obs = self._get_observation()
        # TODO remove later
        assert self.observation_space.contains(np.array(obs))
        return obs

    def _controlled_reset(self, state):
        """
        Perform controlled reset
        """
        # set state and check for collision
        self.environment.setState(torch.Tensor(state))

    def step(self, action):
        """
        Perform step, nothing happens if action is illegal
        """
        # clip action
        action = np.clip(
            action,
            -self.max_action,
            self.max_action
        )

        # 2 steps with half the action
        action = action / 2
        for _ in range(2):
            # Set candidate next state and simulate
            old_state = self.environment.getState()
            state = old_state[1:] + action
            # clip to legal range
            state = np.clip(
                state,
                self.state_3d_low[1:],
                self.state_3d_high[1:]
            )
            action_now = state - old_state[1:]

            self.environment.simulateStep(list(action_now))

        # Renew watch() if render was activated
        if self.watch:
            self.environment.watch()

        # Get observation
        obs = self._get_observation()

        # Calculate reward based on door opening
        door_opening_var = self.environment.getState()[0]
        is_open = door_opening_var < -0.3
        reward = float(is_open)

        done = False
        info = {
            "is_success": is_open,
            "distance": door_opening_var - (-0.3)
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Create view of env
        """
        self.watch=True
        self.environment.watch()

    def _get_observation(self):
        """
        Helper function to return current observation
        """
        if not self.observe_3d_state:
            # always get image
            with torch.no_grad():
                I, M, KT, _, _ = self.environment.getObservation()
                M = M[:,:,[0,2]] # mask 0 is door (including the knob), mask 2 pusher
                if self.perturb_masks is not None:
                    M = self.perturbMasks(
                        M,
                        self.perturb_masks["nH"],
                        self.perturb_masks["sH"]
                    )

            # now, the encoding depends on self.encoder_model
            if self.encoder_model == "Comp3DDyn":
                # Get Nerf encoding for resulting state
                with torch.no_grad():
                    z = self.comp3DDyn.objectEncoder(
                        I=I.to(self.device),
                        M=M.to(self.device),
                        KT=KT.to(self.device)
                    )  # B, nM, k;  1, 2, k
                    # gym requires numpy arrays
                    return z.detach().to('cpu').flatten().numpy()
            elif self.encoder_model == "CNNCNN":
                # Get CNN encoding for resulting state
                with torch.no_grad():
                    if self.exp_config['global']:
                        M = (torch.sum(M, dim=2, keepdim=True) > 0)*1.0  # B, V, 1, H, W
                    z = self.cnncnn.objectEncoder(
                        I=I.to(self.device),
                        M=M.to(self.device),
                        KT=KT.to(self.device)
                    )  # B, nM, k;  1, nM, k;   nM = 1 if global is True
                    return z.detach().to('cpu').flatten().numpy()
            elif self.encoder_model == "CNNNeRF":
                # Get CNN encoding for resulting state
                with torch.no_grad():
                    if self.exp_config['global']:
                        M = (torch.sum(M, dim=2, keepdim=True) > 0)*1.0  # B, V, 1, H, W
                    z = self.cnnnerf.objectEncoder(
                        I=I.to(self.device),
                        M=M.to(self.device),
                        KT=KT.to(self.device)
                    )  # B, nM, k;  1, nM, k;   nM = 1 if global is True
                    return z.detach().to('cpu').flatten().numpy()
            elif self.encoder_model == "ConPre":
                # CURL encoding
                with torch.no_grad():
                    I = I.to(self.device)
                    I = I[:, self.curl.C['viewIndex']]
                    IQuery = centerCrop(I, self.curl.C['HC'], self.curl.C['WC'])  # B, C, H, W
                    B = IQuery.shape[0]
                    nM = 1
                    z = self.curl.objectEncoder(IQuery).view(B, nM, self.curl.C['k'])
                    return z.detach().to('cpu').flatten().numpy()
            elif self.encoder_model == "TimeConPre":
                # CURL time cont encoding
                with torch.no_grad():
                    z = self.curltime.objectEncoder(
                        I=I.to(self.device),
                        M=M.to(self.device),
                        KT=KT.to(self.device)
                    )  # B, nM, k;  1, nM, k;   nM = 1 if global is True
                    return z.detach().to('cpu').flatten().numpy()

            else:
                raise Exception("Unknown encoder_model")

        if self.observe_3d_state:    
            if self.use_keypoints:
                return self.environment.getKeypoints().flatten()
            elif self.use_fourier:
                return self.environment.getHighDimState().flatten()
            else:
                # Get 3D state
                return self.environment.getState().detach().to('cpu').numpy()

    def perturbMasks(self, M, nH, sH):
        """
        :param M: B, V, nM, H, W
        :param nH: number of box perturbations
        :param sH: size of box perturbation in each direction
        :return: B, V, nM, H, W
        """
        for maskBatch in M:
            for maskView in maskBatch:
                for maskLatent in maskView:
                    a = torch.nonzero(maskLatent == 1)
                    if a.shape[0] == 0: continue
                    inds = np.random.randint(0, a.shape[0], size=nH)
                    a = a[inds]
                    for b in a:
                        maskLatent[b[0] - sH:b[0] + sH, b[1] - sH:b[1] + sH] = 0.
        return M
