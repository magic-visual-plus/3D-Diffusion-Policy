from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from loguru import logger 
import zarr
import numpy as np
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
import random

def test_replay_buffer_load():
    dataset_path = '/opt/projects/3D-Diffusion-Policy/data/realdex_pour.zarr'
    replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=['state', 'action', 'point_cloud', 'img'])
    logger.info("chunk size {}", replay_buffer.chunk_size)
    logger.info("n_episodes {}", replay_buffer.n_episodes)
    result = replay_buffer.get_episode(0)
    logger.info("type result {}", result)

    
def test_sampler_load():
    dataset_path = '/opt/projects/3D-Diffusion-Policy/data/realdex_pour.zarr'
    replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=['state', 'action', 'point_cloud', 'img'])
    logger.info("chunk size {}", replay_buffer.chunk_size)
    logger.info("n_episodes {}", replay_buffer.n_episodes)
    # logger.info("episode 0 {}", replay_buffer.get_episode(0))
    episode_mask = np.zeros(replay_buffer.n_episodes)
    episode_mask[0] = 1
    action_step = 8
    sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=16,
            pad_before=1, 
            pad_after=action_step-1,
            episode_mask=episode_mask)
    logger.info("sampler len {}", len(sampler))
    idx = random.randint(0, len(sampler))
    logger.info("sample idx {}", idx)
    result = sampler.sample_sequence(idx)
    logger.info("result keys {}", result.keys())
    
    for idx in range(0, len(sampler), action_step//2-1):
        result = sampler.sample_sequence(idx)
        logger.info("idx {} state {}", idx, result['state'])

if __name__ == '__main__':
#     test_replay_buffer_load()
    test_sampler_load()