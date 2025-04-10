# deploy model to real device
from loguru import logger
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
import zarr
import numpy as np
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
import random
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import time
from profiler import Dbg_Timer
import matplotlib.pyplot as plt



OmegaConf.register_new_resolver("eval", eval, replace=True)
cur_dir = os.path.dirname(os.path.abspath(__file__))


class BaseInferenceClient():
    
    def step(self, action_list):
        ...

    def reset(self, first_init=True) -> dict:
        ...
        

class AuboSimEnvInferenceClient(BaseInferenceClient):
    """从相机里面或者训练的数据集里面读取obs，发送给仿真的aubo臂，进行模型控制运动
    """
    def __init__(self, obs_horizon=2, action_horizon=8, sequence_length=16, device="gpu",
                use_point_cloud=True, use_image=True, img_size=84,
                 num_points=4096, sim_mode='dataset', sim_episode_idx = 0,
                 use_waist=False):
        # 需要给模型的obs长度
        self.obs_horizon = obs_horizon
        # 预测的给臂运动的长度
        self.action_horizon = action_horizon
        self.sequence_length = sequence_length
        # 预测设备id
        self.device = device
        # 预测模式，支持从本地数据集或者从真机设备采集
        self.sim_mode = sim_mode
        self.sim_episode_idx = sim_episode_idx
        
    def init(self):
        dataset_path = '/opt/projects/3D-Diffusion-Policy/data/realdex_pour.zarr'
        # dataset_path = '/opt/projects/3D-Diffusion-Policy/data/aubo.zarr'
        # dataset_path = '/opt/projects/3D-Diffusion-Policy/data/aubo_delta.zarr'
        dataset_path = '/mnt/d/3.0/3D-Diffusion-Policy/data/realdex_pour.zarr'
        if self.sim_mode == 'dataset':
            replay_buffer = ReplayBuffer.copy_from_path(
                    dataset_path, keys=['state', 'action', 'point_cloud', 'img'])
            logger.info("chunk size {}", replay_buffer.chunk_size)
            logger.info("n_episodes {}", replay_buffer.n_episodes)
            
            episode_mask = np.zeros(replay_buffer.n_episodes)
            # just need one episode to test
            episode_mask[self.sim_episode_idx] = 1
            sampler = SequenceSampler(
                replay_buffer=replay_buffer, 
                sequence_length=self.sequence_length,
                pad_before=self.obs_horizon - 1, 
                pad_after=self.action_horizon - 1,
                episode_mask=episode_mask)
            # logger.info("sampler len {}", len(sampler))
            self.sampler = sampler
            self.sample_data_idxs = [i for i in range(0, len(sampler), self.action_horizon-1)]
            self.sample_current_idx = 0
            self.sample_data_len = len(sampler)
            logger.info("sample data idxs {}, current idx {}, sample data len {}", self.sample_data_idxs, self.sample_current_idx, self.sample_data_len)
        else:
            # load env from sim
            ...
        
    def step(self, action_list):
        if self.sim_mode == 'dataset':
            self.sample_current_idx += 1
            # just return next obs
            sample_idx = self.sample_data_idxs[self.sample_current_idx]
            obs_dict = self.sampler.sample_sequence(sample_idx)
            obs_dict['agent_pos'] = obs_dict['state']
            return obs_dict
        else:
            ...
        
    def reset(self, first_init=True) -> dict:
         # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
        
        if self.sim_mode == 'dataset':
            self.sample_current_idx = 0
            # todo reset robot env
            obs_dict = self.sampler.sample_sequence(self.sample_current_idx)
            obs_dict['agent_pos'] = obs_dict['state']
            return obs_dict
            # np_obs_dict = {'point_cloud': obs_dict['point_cloud'], 'agent_pos': obs_dict['agent_pos']}
            # return np_obs_dict
        else:
            ...



class AuboEnvInferenceClient(BaseInferenceClient):
    ...
    
    def step(self, action_list):
        """execute current action, return obs 

        Args:
            action_list (_type_): _description_
        """
        ...
        
    def reset(self, first_init=True) -> dict:
        """return current obs dict

        Args:
            init_first (bool, optional): _description_. Defaults to True.

        Returns:
            dict: _description_
        """
        ...
        


def online_infer(env: BaseInferenceClient, policy: BasePolicy):
    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 10,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 84
    num_points = 4096
    use_waist = True
    first_init = True
    record_data = True
    
    # reset env first to get robot pos
    obs_dict = env.reset(first_init=first_init)
    logger.info("obs dict keys {}", obs_dict.keys())

    step_count = 0
    action_horizon = policy.n_action_steps
    device = policy.device
    plot_traj = True
    delta_action = True
    
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []
    
    while step_count < roll_out_length:
        state_joints_across_time.extend(obs_dict['agent_pos'][:action_horizon,].tolist())
        gt_action_joints_across_time.extend(obs_dict['action'][:action_horizon,].tolist())
            
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        with torch.no_grad():
            # run mode info predict next action
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            print(f"agent_pos shape {obs_dict_input['agent_pos'].shape}, point_cloud shape {obs_dict_input['point_cloud'].shape}")
            with Dbg_Timer(f"predict_one_action_{step_count}"):
                action_dict = policy.predict_action(obs_dict_input)
            # device_transfer, move to cpu 
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
            action_list = np_action_dict['action'].squeeze(0)
            # logger.info("step {} expect action {}", step_count, obs_dict['action'])
            # logger.info("step {} infer action {}", step_count, action_list)
            pred_action_joints_across_time.extend(action_list.tolist())
            
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        logger.info("step count {}", step_count)
        
    if plot_traj:
        # plot traj
        state_joints_across_time = np.array(state_joints_across_time)
        gt_action_joints_across_time = np.array(gt_action_joints_across_time)
        pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:len(state_joints_across_time)]
        # 
        
        assert (
            state_joints_across_time.shape
            == gt_action_joints_across_time.shape
            == pred_action_joints_across_time.shape
        )
        if delta_action:
            gt_action_joints_across_time = state_joints_across_time + gt_action_joints_across_time
            pred_action_joints_across_time = state_joints_across_time + pred_action_joints_across_time
        steps = len(state_joints_across_time)
        # calc MSE across time
        mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
        print("Unnormalized Action MSE across single traj:", mse)
        num_of_joints = state_joints_across_time.shape[1]
        
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))
        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory",
            fontsize=16,
            color="blue",
        )
        for i, ax in enumerate(axes):
            ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action joints")
            # put a dot every ACTION_HORIZON
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")
            ax.set_title(f"Joint {i}")
            ax.legend()
        plt.tight_layout()
        traj_snapshot_path = f'{cur_dir}/trajectory_plot.png'
        logger.info("save traj to dir {}", traj_snapshot_path)
        plt.savefig(traj_snapshot_path)
        plt.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    print(f'cfg {cfg}, type {type(cfg)}')
    with Dbg_Timer("init env"):
        env = AuboSimEnvInferenceClient()
        env.init()
    # infer policy model
    base_dir = '/opt/projects/data/outputs'
    with Dbg_Timer("init policy"):
        workspace = TrainDP3Workspace(cfg)
        policy = workspace.load_model()
    with Dbg_Timer("run online_infer"):
        online_infer(env, policy)
    

def test_infer():
    env = AuboSimEnvInferenceClient()
    env.init()
    ...
    
if __name__ == '__main__':
    main()
    # test_infer()