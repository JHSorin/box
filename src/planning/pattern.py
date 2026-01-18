import numpy as np
import yaml
from planning.packer import GridPacker

class Pattern:
    def __init__(self, pallet_config_path='config/pallet_config.yaml', box_config_path='config/box_config.yaml'):
        with open(pallet_config_path, 'r') as f:
            self.pallet_config = yaml.safe_load(f)
        with open(box_config_path, 'r') as f:
            self.box_config = yaml.safe_load(f)
            
        self.pallet_dims = self.pallet_config['pallet_size'] # [L, W, H]
        self.box_dims = self.box_config['dimensions'] # [L, W, H]
        
        # Optimize: Use resolution from config (e.g., 20mm) instead of hardcoded 1mm
        # This reduces grid size from 2000x2000 (4M points) to 100x100 (10k points),
        # speeding up integral image calculation by ~400x while maintaining sufficient accuracy.
        resolution = self.pallet_config.get('grid_resolution', 20)
        self.packer = GridPacker(self.pallet_dims, resolution=resolution)
        
        # FORCE OBSTACLES: Place random obstacles to test robustness
        # 1. Center Obstacle (Original)
        cx, cy = self.pallet_dims[0] / 2, self.pallet_dims[1] / 2
        self.packer.force_place_box(cx, cy, self.box_dims, theta=45)
        print(f"Pattern: Forced obstacle at ({cx}, {cy}) with 45 degrees.")
        
        # 2. Corner Obstacle (Top-Right)
        # Avoid exact edge to not be out of bounds
        ox2 = self.pallet_dims[0] - 300
        oy2 = self.pallet_dims[1] - 300
        self.packer.force_place_box(ox2, oy2, self.box_dims, theta=30)
        print(f"Pattern: Forced obstacle at ({ox2}, {oy2}) with 30 degrees.")
        
        # 3. Edge Obstacle (Bottom-Middle-Left)
        ox3 = 400
        oy3 = 100 # Close to edge
        self.packer.force_place_box(ox3, oy3, self.box_dims, theta=15)
        print(f"Pattern: Forced obstacle at ({ox3}, {oy3}) with 15 degrees.")
        
        # 4. Another random one
        ox4 = 1500
        oy4 = 600
        self.packer.force_place_box(ox4, oy4, self.box_dims, theta=60)
        print(f"Pattern: Forced obstacle at ({ox4}, {oy4}) with 60 degrees.")
        
        # Calculate optimal pattern around the obstacle
        self.optimal_pattern = self.packer.pack(self.box_dims)
        print(f"Pattern Optimized: Found layout with {len(self.optimal_pattern)} boxes (including obstacle) using Grid Search.")

    def get_pattern(self):
        """
        Return the optimal pattern for the single layer.
        """
        return self.optimal_pattern
