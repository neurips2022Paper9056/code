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
import pushingEnvironment

class BoxEnvNerf(gym.Env):
    """
    Gym env wrapper for PushingEnvironment
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        exp_path,
        network_number,
        device,
        encoder_model,
        reset_int=None,
        observe_3d_state=False,
        use_keypoints=False,
        use_fourier=False,
        perturb_masks=None,
    ):
        self.device = device
        self.encoder_model = encoder_model
        self.reset_int = reset_int
        self.observe_3d_state = observe_3d_state
        self.use_keypoints = use_keypoints
        self.use_fourier = use_fourier
        self.perturb_masks=perturb_masks

        if self.observe_3d_state:
            assert self.reset_int is not None, "Need constant shape for 3D state (otherwise MDP is ill-posed)"

        # this is redundant but doesn't hurt
        if self.use_keypoints:
            assert self.reset_int is not None, "Need constant shape for keypoints (otherwise MDP is ill-posed due to color)"

        # this is redundant but doesn't hurt
        if self.use_fourier:
            assert self.reset_int is not None, "Need constant shape for fourier (otherwise MDP is ill-posed due to color)"

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
            elif self.encoder_model == "TimeConPre":
                self.curltime = CURL.TimeConPre(**self.exp_config)
                self.curltime.buildModel()
                self.curltime.loadNetwork(exp_path + '/' + str(network_number))
            elif self.encoder_model == "ConPre":
                self.curl = CURLSingleView.ConPre(**self.exp_config)
                self.curl.buildModel()
                self.curl.loadNetwork(exp_path + '/' + str(network_number))
                self.curl.objectEncoder.eval() # not important here, but better be safe
                if self.curl.multipleLatentVectors:
                    raise NotImplementedError
            else:
                raise Exception("Unknown encoder_model")

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
                    low=-1, high=-1,
                    shape=obs_shape
                )
            elif self.use_fourier:
                obs_shape = (160,)
                self.observation_space = gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=obs_shape
                )
            else:
                # Get 3D state
                self.observation_space = gym.spaces.Box(
                    low=np.array([-0.2, -0.2, -1.0, -1.0, -1, -1, -1, -1]),
                    high=np.array([0.2, 0.2, 1.0, 1.0, 1, 1, 1, 1])
                )
                

        if self.perturb_masks is not None:
            assert not self.observe_3d_state, "Mask perturbation has no effect on 3D state"

        # determine action space
        self.action_space = gym.spaces.Box(
            low=np.array([-0.02, -0.02]),
            high=np.array([0.02, 0.02])
        )

        # init
        self.environment = None
        self.current_color = None
        self.watch = False
        self.reset()


    def reset(self):
        """
        Reset (either to random position or to the same)
        """
        if self.reset_int is not None:
            pos_reset_int = np.random.randint(1000000000)

        return self._controlled_reset(
            self.reset_int if self.reset_int is not None else np.random.randint(1000000000),
            np.random.randint(1000000000) if self.reset_int is not None else -1
        )


    def _controlled_reset(self, reset_int, pos_reset_ind):
        """
        Reset to specific seed
        """
        # reset
        self.environment = pushingEnvironment.PushingEnvironment()
        self.current_color = self.environment.initEnvironment(
            reset_int,
            pos_reset_ind
        )

        # Renew watch() if render was activated
        if self.watch:
            self.environment.watch()

        # create simulation
        self.environment.initSimulation()

        # return observation
        return self._get_observation()


    def step(self, action):
        """
        Perform env step
        """
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        current_pos = np.array(self.environment.getState()[:2])
        next_pos = np.clip(
            current_pos + action,
            [-0.16, -0.16],
            [0.16, 0.16]
        )
        # action = torch.Tensor(next_pos-current_pos).to(self.device)
        action = list(next_pos-current_pos)
        self.environment.simulateStep(action)

        # Renew watch() if render was activated
        if self.watch:
            self.environment.watch()

        # Get observation
        next_obs = self._get_observation()

        # Get reward
        if self.current_color == 0:
            # blue
            reward = float(
                float(self.environment.getState()[2]) > 0.12
            )
        elif self.current_color == 1:
            # yellow
            reward = float(
                float(self.environment.getState()[2]) < -0.12
            )

        # get "done"
        done = False

        # get info
        info = {
            "is_success": reward > 0,
            "distance": 42. #TODO
        }

        return next_obs, reward, done, info

    def _get_observation(self):
        """
        Helper function to obtain current obs
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
                    obs = z.detach().to('cpu').flatten().numpy()
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
                    obs = z.detach().to('cpu').flatten().numpy()
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
                    obs = z.detach().to('cpu').flatten().numpy()
            elif self.encoder_model == "TimeConPre":
                # CURL time cont encoding
                with torch.no_grad():
                    z = self.curltime.objectEncoder(
                        I=I.to(self.device),
                        M=M.to(self.device),
                        KT=KT.to(self.device)
                    )  # B, nM, k;  1, nM, k;   nM = 1 if global is True
                    obs = z.detach().to('cpu').flatten().numpy()

            else:
                raise Exception("Unknown encoder_model")

        if self.observe_3d_state:
            if self.use_keypoints:
                obs = self.environment.getKeypoints().flatten()
            elif self.use_fourier:
                obs = self.environment.getHighDimState().flatten()
            else:
                # Get 3D state
                obs = self.environment.getState().numpy()
                
        return obs

    def render(self, mode='human'):
        """
        Create view of env
        """
        self.watch=True
        self.environment.watch()

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
