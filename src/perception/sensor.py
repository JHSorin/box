import numpy as np
import os
import cv2

class Sensor:
    def __init__(self, mode='real', data_dir='data/samples', virtual_source=None):
        self.mode = mode
        self.data_dir = data_dir
        self.current_sample_index = 0
        self.virtual_source = virtual_source # Reference to VirtualPallet

    def get_rgbd(self):
        """
        Get RGB and Depth images.
        Returns:
            rgb_image (numpy array)
            depth_image (numpy array)
        """
        if self.mode == 'offline':
            return self._get_offline_data()
        elif self.mode == 'simulation':
             return self._get_simulation_data()
        else:
            return self._get_real_data()

    def _get_simulation_data(self):
        if self.virtual_source:
            depth = self.virtual_source.get_depth_image()
            rgb = self.virtual_source.get_rgb_image()
            # Ensure depth is transposed if needed to match RGB?
            # In VirtualPallet, get_depth returns (W, H). get_rgb returns (H, W, 3) (standard image)
            # We should transpose depth to match image (H, W)
            return rgb, depth.T
        else:
            raise ValueError("Virtual source not configured for simulation mode")

    def _get_offline_data(self):
        # Existing mock implementation...
        print(f"Reading offline data sample {self.current_sample_index}...")
        rgb = np.ones((480, 640, 3), dtype=np.uint8) * 255
        depth = np.ones((480, 640), dtype=np.uint16) * 1000 
        depth[200:300, 250:350] = 700
        self.current_sample_index += 1
        return rgb, depth

    def _get_real_data(self):
        raise NotImplementedError("Real sensor interface not implemented yet.")
