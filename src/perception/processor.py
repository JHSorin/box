import numpy as np
import yaml

class Processor:
    def __init__(self, config_path='config/pallet_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.resolution = self.config['grid_resolution']
        self.pallet_size = self.config['pallet_size']
        
        # Grid size
        self.grid_w = int(self.pallet_size[0] / self.resolution)
        self.grid_h = int(self.pallet_size[1] / self.resolution)
        self.camera_z = self.config.get('camera_height', 2000)

    def build_height_map(self, depth_image):
        """
        Convert depth image to height map.
        """
        
        # Cast to float32 to avoid uint16 underflow when subtracting
        depth_float = depth_image.astype(np.float32)
        
        height_map_img = self.camera_z - depth_float
        height_map_img = np.maximum(height_map_img, 0) 
        
        # Transpose back to get [X, Y] grid
        height_map = height_map_img.T
        
        return height_map
