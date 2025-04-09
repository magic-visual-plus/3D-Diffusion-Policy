from diffusion_policy_3d.dataset.realdex_dataset import RealDexDataset
from loguru import logger
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)



def test_read_real_dex_dataset():
    dataset_path = '/opt/projects/3D-Diffusion-Policy/data/realdex_pour.zarr'
    dataset = RealDexDataset(dataset_path, horizon=16, pad_before=1, pad_after=7, max_train_episodes=90)
    logger.info("dataset len {}", len(dataset))
    idx = 0
    result = dataset[idx]
    logger.info("result {}", result)
    
if __name__ == '__main__':
    test_read_real_dex_dataset()