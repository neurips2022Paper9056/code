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

sys.path.append(os.getenv("HOME") + '/git/imorl/projects/mugHangingNeRF')
import mugEnvironment

class MugEnvNerf(gym.Env):
    """
    Gym env wrapper for MugEnvironment
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        exp_path,
        network_number,
        device,
        encoder_model,
        new_mug_every_n_resets=1, # None corresponds to never
        reset_int=None, # None corresponds to random choice at each reset
        reset_position=None,
        observe_3d_state=False,
        use_keypoints=False,
        use_fourier=False,
        perturb_masks=None,
        max_action=0.01
    ):
        self.device = device
        self.encoder_model = encoder_model
        self.new_mug_every_n_resets = new_mug_every_n_resets
        self.reset_int = reset_int
        self.reset_position=torch.Tensor(reset_position) if reset_position is not None else None
        self.observe_3d_state=observe_3d_state
        self.use_keypoints=use_keypoints
        self.use_fourier = use_fourier
        self.perturb_masks=perturb_masks
        self.max_action = max_action


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
                self.exp_config = self.cnnnerf.C
                if not 'global' in self.exp_config:
                    self.exp_config['global']=False
            elif self.encoder_model == "ConPre":
                self.curl = CURLSingleView.ConPre(**self.exp_config)
                self.curl.buildModel()
                self.curl.loadNetwork(exp_path + '/' + str(network_number))
                self.curl.objectEncoder.eval() # not important here, but better be safe
                if self.curl.multipleLatentVectors:
                    raise NotImplementedError
            elif self.encoder_model == "TimeConPre":
                self.curltime = CURL.TimeConPre(**self.exp_config)
                self.curltime.buildModel()
                self.curltime.loadNetwork(exp_path + '/' + str(network_number))
            else:
                raise Exception("Unknown encoder_model")

        if self.new_mug_every_n_resets is None:
            assert self.reset_int is not None, "Mug shape must be configured if it is not random"
        else:
            assert self.reset_int is None, "You configured random mug resets AND a mug shape"

        if self.observe_3d_state and not self.use_keypoints:
            assert self.reset_int is not None, "Need constant shape for 3D state (otherwise MDP is ill-posed)"

        if self.perturb_masks is not None:
            assert not self.observe_3d_state, "Mask perturbation has no effect on 3D state"

        if self.use_fourier:
            assert self.reset_int is not None, "Need constant shape for fourier (otherwise MDP is ill-posed due to color)"

        self.environment = mugEnvironment.MugEnvironment()
        self.watch = False
        self.environment.initEnvironment(
            self.reset_int if self.reset_int is not None else np.random.randint(1000000000)
        )

        tmp = self.environment.getBoundingBoxLimits()
        self.box_low = torch.Tensor(tmp[::2])
        self.box_high = torch.Tensor(tmp[1::2])

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
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=obs_shape
            )

        if self.observe_3d_state:
            if self.use_keypoints:
                obs_shape = (12,)
                self.observation_space = gym.spaces.Box(
                    low=0, high=0.69,
                    shape=obs_shape
                )
            elif self.use_fourier:
                obs_shape = (60,)
                self.observation_space = gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=obs_shape
                )
            else:
                # Get 3D state
                obs_shape = (3,)
                self.observation_space = gym.spaces.Box(
                    low=0, high=0.69,
                    shape=obs_shape
                )

        # determine action space
        self.action_space = gym.spaces.Box(
            low=-max_action*np.ones(3),
            high=max_action*np.ones(3)
        )

        self.n_resets = 0
        self.reset()

    def reset(self):
        """
        Reset (currently to fixed position but) with random mug size
        """
        # reset mug shape
        if self.new_mug_every_n_resets is not None:
            if self.n_resets % self.new_mug_every_n_resets==0:
                self.environment.initEnvironment(
                    self.reset_int if self.reset_int is not None else np.random.randint(1000000000)
                )

        # and increment self.n_resets
        self.n_resets += 1

        if self.reset_position is None:
            # randomize initial position
            while True:
                # sample state
                state = torch.rand(len(self.box_low)) * (
                    self.box_high - self.box_low
                ) + self.box_low
                if self._controlled_reset(state, handle_not_valid="return"):
                    break
        else:
            self._controlled_reset(self.reset_position, handle_not_valid="error")

        # Renew watch() if render was activated
        if self.watch:
            self.environment.watch()

        # return observation
        return self._get_observation()

    def _controlled_reset(self, state, handle_not_valid="error"):
        """
        Perform controlled reset
        """
        # set state and check for collision
        self.environment.setState(state)
        _, colliding = self.environment.checkStateInSim()

        # Sample again if...
        valid = not (
            # ...mug is outside of boundary box or
            (any(state < self.box_low) or any(state > self.box_high))
            or
            # ...mug is in collision
            colliding
        )

        if handle_not_valid=="error":
            assert valid, "Reset position is not valid (outside bounding box or in collision)"
        elif handle_not_valid=="return":
            return valid
        else:
            raise Exception("Invalid argument for handle_not_valid")

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

        # Set candidate next state and simulate
        old_state = self.environment.getState()
        state = old_state + action
        self.environment.setState(state)
        stable, colliding = self.environment.checkStateInSim()

        # Nothing happens if...
        if (
            # ...mug is outside of boundary box or
            (any(state < self.box_low) or any(state > self.box_high))
            or
            # ...mug is in collision
            colliding
        ):
            self.environment.setState(old_state)
            # simulate for old_state and overwrite
            stable, colliding = self.environment.checkStateInSim()

        # Renew watch() if render was activated
        if self.watch:
            self.environment.watch()

        # Get observation
        obs = self._get_observation()

        # Calculate reward based on stable-ness
        # (already computed above)
        reward = float(stable)

        done = False
        info = {
            "is_success": stable,
            "distance": 42. #TODO
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
            if self.use_fourier:
                return self.environment.getHighDimState().flatten()
            if self.use_keypoints:
                return self.environment.getKeypoints().flatten()
            return self.environment.getState()

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
