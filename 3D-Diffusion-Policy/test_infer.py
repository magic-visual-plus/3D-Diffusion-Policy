from loguru import logger
from network import ZMQResponseServer, ZMQResponseClient
import numpy as np


def mock_data():
    obs_dict = {}
    obs_dict['point_cloud'] = np.random.randint(256, size=(2, 1024, 6), dtype=np.uint8)
    obs_dict['state'] = np.random.randint(256, size=(2, 7), dtype=np.uint8)
    return obs_dict


def test_infer():
    # zmq_client = ZMQResponseClient('172.23.255.227', 8000)
    # zmq_client = ZMQResponseClient('192.168.110.200', 8899)
    zmq_client = ZMQResponseClient('192.168.110.200', 8000)
    obs_dict = mock_data()
    action_result = zmq_client.send_request(obs_dict)
    logger.info("action result {}", action_result)
    zmq_client.stop()
    
if __name__ == '__main__':
    test_infer()