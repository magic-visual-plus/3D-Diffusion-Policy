# deploy model to real device
from loguru import logger
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
import numpy as np
import random
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.base_policy import BasePolicy
import time
from profiler import Dbg_Timer
from network import ZMQResponseServer, ZMQResponseClient


OmegaConf.register_new_resolver("eval", eval, replace=True)
cur_dir = os.path.dirname(os.path.abspath(__file__))

def servo_infer(policy: BasePolicy):
    step_count = 0
    device = policy.device
    zmq_server = ZMQResponseServer("0.0.0.0", 8000)
    logger.info("init server done")
    while True:
        obs_dict = zmq_server.recv_request()
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        with torch.no_grad():
            # run mode info predict next action
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['state'].unsqueeze(0)
            print(f"agent_pos shape {obs_dict_input['agent_pos'].shape}, point_cloud shape {obs_dict_input['point_cloud'].shape}")
            action_list = []
            try:
                with Dbg_Timer(f"predict_one_action_{step_count}"):
                    action_dict = policy.predict_action(obs_dict_input)
            except Exception as e:
                logger.exception(e)
            # device_transfer, move to cpu 
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
            action_list = np_action_dict['action'].squeeze(0)
            logger.info("step {} infer action {}", step_count, action_list)
        obs_result = {"action": action_list}
        zmq_server.send_response(obs_result)

        

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    print(f'cfg {cfg}, type {type(cfg)}')
    with Dbg_Timer("init policy"):
        workspace = TrainDP3Workspace(cfg)
        policy = workspace.load_model()
    with Dbg_Timer("run online_infer"):
        servo_infer(policy)
    
    
if __name__ == '__main__':
    main()