import numpy as np
import yaml
import cv2

class Solver:
    def __init__(self, config_path='config/pallet_config.yaml', box_config_path='config/box_config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.resolution = config['grid_resolution']
        self.pallet_size = config['pallet_size']
        self.grid_w = int(self.pallet_size[0] / self.resolution)
        self.grid_h = int(self.pallet_size[1] / self.resolution)
        
        with open(box_config_path, 'r') as f:
             box_config = yaml.safe_load(f)
        self.box_dims = box_config['dimensions'] # L, W, H

    def find_slot(self, height_map, pattern_slots, target_z_height=0):
        """
        Find the next valid slot from the pattern based on the height map.
        Uses precise polygon masking to calculate average height under the box.
        """
        
        epsilon = 50 # Tolerance in mm
        
        print(f"DEBUG: find_slot called. TargetZ={target_z_height}. PatternSlots={len(pattern_slots)}")
        
        for i, slot in enumerate(pattern_slots):
            # Use mask to check exact footprint
            avg_height = self._get_masked_average_height(height_map, slot['x'], slot['y'], slot['theta'])
            
            if avg_height is None:
                # print(f"DEBUG: Slot {i} ({slot['x']}, {slot['y']}) Skipped (Out of bounds or invalid mask)")
                continue
            
            # For single layer, check if the slot is occupied (height > epsilon)
            is_filled = avg_height > (target_z_height + epsilon)
            
            # Uncommented Debug Print
            print(f"DEBUG: Slot {i} ({slot['x']}, {slot['y']}, {slot['theta']}) AvgH={avg_height:.1f} Filled={is_filled}")
            
            if not is_filled:
                # Return the slot with Z height added
                result = slot.copy()
                result['z'] = target_z_height
                return result
                    
        return None

    def _get_masked_average_height(self, height_map, cx, cy, theta_deg):
        # Calculate bounding box to extract small ROI first (optimization)
        L = self.box_dims[0]
        W = self.box_dims[1]
        
        # Grid Indices
        cx_idx = cx / self.resolution
        cy_idx = cy / self.resolution
        L_idx = L / self.resolution
        W_idx = W / self.resolution
        
        theta_rad = np.radians(theta_deg)
        
        # Calculate AABB for slicing
        extent_x = L_idx * abs(np.cos(theta_rad)) + W_idx * abs(np.sin(theta_rad))
        extent_y = L_idx * abs(np.sin(theta_rad)) + W_idx * abs(np.cos(theta_rad))
        
        half_x = int(extent_x / 2) + 2 # Add padding
        half_y = int(extent_y / 2) + 2
        
        cxi = int(cx_idx)
        cyi = int(cy_idx)
        
        x_start = max(0, cxi - half_x)
        x_end = min(self.grid_w, cxi + half_x)
        y_start = max(0, cyi - half_y)
        y_end = min(self.grid_h, cyi + half_y)
        
        if x_start >= x_end or y_start >= y_end:
            return None
            
        # Extract sub-grid
        roi_height = height_map[x_start:x_end, y_start:y_end]
        
        # Create mask on sub-grid coordinate system
        # Local Center
        lcx = cx_idx - x_start
        lcy = cy_idx - y_start
        
        # Box corners
        # Shrink mask slightly to avoid boundary overlap
        shrink_factor = 0.95
        hw = (L_idx * shrink_factor) / 2.0
        hh = (W_idx * shrink_factor) / 2.0
        corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        corners_rot = np.dot(corners_local, rot_matrix.T)
        corners_global = corners_rot + np.array([lcx, lcy])
        
        pts = corners_global.astype(np.int32)
        pts_cv = np.zeros_like(pts)
        pts_cv[:, 0] = pts[:, 1] # Col = y (in local mask)
        pts_cv[:, 1] = pts[:, 0] # Row = x (in local mask)
        
        # Mask shape matches ROI shape
        mask = np.zeros(roi_height.shape, dtype=np.uint8)
        
        # OpenCV expects (Col, Row) points
        cv2.fillPoly(mask, [pts_cv], 1)
        
        masked_values = roi_height[mask == 1]
        
        if masked_values.size == 0:
            return 0.0 # Should not happen if box is valid
            
        return np.mean(masked_values)
