import numpy as np
import yaml
import cv2

class VirtualPallet:
    def __init__(self, config_path='config/pallet_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pallet_size = self.config['pallet_size'] # [L, W, H]
        self.resolution = self.config['grid_resolution']
        
        # Initialize height map (grid)
        self.grid_w = int(self.pallet_size[0] / self.resolution)
        self.grid_h = int(self.pallet_size[1] / self.resolution)
        self.height_map = np.zeros((self.grid_w, self.grid_h), dtype=np.float32)
        self.camera_z = self.config.get('camera_height', 2000)
        
        self.boxes = [] # Keep track of placed boxes for visualization
        self.cached_bg = None
        self.cached_bg_layer = None
        self.current_active_layer = 0

    def set_active_layer(self, layer_index):
        if self.current_active_layer != layer_index:
            self.cached_bg = None # Invalidate cache if layer changes
        self.current_active_layer = layer_index

    def add_box(self, x, y, width, length, height, theta=0, is_obstacle=False, layer_index=0):
        """
        Place a box on the virtual pallet.
        Supports arbitrary rotation using rasterization.
        """
        self.boxes.append({
            'x': x, 'y': y, 
            'w': width, 'l': length, 'h': height, 
            'theta': theta,
            'is_obstacle': is_obstacle,
            'layer': layer_index
        })
        
        # Rasterize on grid
        self._rasterize_box_on_heightmap(x, y, length, width, height, theta)
        
        # Update Cache Incrementally
        if self.cached_bg is not None:
             # Recalculate scale
             target_width_px = 1000
             vis_scale = max(0.5, target_width_px / self.grid_w)
             self._draw_box_on_image(self.cached_bg, self.boxes[-1], vis_scale, self.current_active_layer)

    def _rasterize_box_on_heightmap(self, cx, cy, L, W, H, theta_deg):
        # Convert physical coords to grid indices (float)
        cx_idx = cx / self.resolution
        cy_idx = cy / self.resolution
        L_idx = L / self.resolution
        W_idx = W / self.resolution
        
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
      # Box corners
        # Shrink mask slightly to avoid boundary overlap
        shrink_factor = 0.95
        hw = (L_idx * shrink_factor) / 2.0
        hh = (W_idx * shrink_factor) / 2.0
        
        corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corners_rot = np.dot(corners_local, rot_matrix.T)
        corners_global = corners_rot + np.array([cx_idx, cy_idx])
        
        # OpenCV Rasterization
        pts = corners_global.astype(np.int32)
        pts_cv = np.zeros_like(pts)
        pts_cv[:, 0] = pts[:, 1] # Col = y
        pts_cv[:, 1] = pts[:, 0] # Row = x
        
        # Create mask for this box
        # Fix: Use grid_h instead of grid_l
        mask = np.zeros((self.grid_w, self.grid_h), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_cv], 1)
        
        # Update Height Map (Still needed for logic, but we can optimize if only for viz)
        # But wait, find_slot uses height_map! So we MUST keep height_map logic.
        masked_region = self.height_map[mask == 1]
        
        if masked_region.size > 0:
            base_height = np.max(masked_region)
            new_height = base_height + H
            
            # Apply new height
            self.height_map[mask == 1] = new_height

    def get_depth_image(self):
        depth_map = self.camera_z - self.height_map
        noise = np.random.normal(0, 2, depth_map.shape)
        depth_map = depth_map + noise
        return depth_map.astype(np.uint16)

    def get_rgb_image(self):
        # Calculate vis_scale dynamically to keep image size manageable (e.g., max 1000px)
        # Pallet size W is self.grid_w (in grid units).
        # We want final width ~ 1000 px.
        target_width_px = 1000
        vis_scale = target_width_px / self.grid_w
        
        # Ensure minimum scale
        vis_scale = max(0.5, vis_scale)
        
        # Determine Pallet Color based on current layer
        current_active_layer = getattr(self, 'current_active_layer', 0)
        
        # Check cache
        if self.cached_bg is None:
            if current_active_layer == 0:
                # Modern Light Palette
                pallet_color = (235, 235, 225) # Off-white/Cream (BGR)
            else:
                pallet_color = (200, 200, 200) # Light Gray - Inactive

            # Create high-res image
            h = int(self.grid_h * vis_scale)
            w = int(self.grid_w * vis_scale)
            self.cached_bg = np.ones((h, w, 3), dtype=np.uint8)
            self.cached_bg[:] = pallet_color
            
            # Draw Background Grid
            # Subtle but visible grid
            grid_spacing_mm = 100
            grid_spacing_px = int((grid_spacing_mm / self.resolution) * vis_scale)
            grid_color = (210, 210, 210) # Light gray for minor grid
            grid_thickness = max(1, int(1 * vis_scale)) 
            
            if grid_spacing_px > 0:
                # Vertical lines
                for x in range(0, w, grid_spacing_px):
                    cv2.line(self.cached_bg, (x, 0), (x, h), grid_color, grid_thickness)
                
                # Horizontal lines
                for y in range(0, h, grid_spacing_px):
                    cv2.line(self.cached_bg, (0, y), (w, y), grid_color, grid_thickness)
            
            # Draw thicker lines every 500mm
            major_grid_spacing = grid_spacing_px * 5
            major_grid_color = (180, 180, 180) # Medium gray for major grid
            major_thickness = max(2, int(2 * vis_scale))
            
            if major_grid_spacing > 0:
                for x in range(0, w, major_grid_spacing):
                    cv2.line(self.cached_bg, (x, 0), (x, h), major_grid_color, major_thickness)
                    
                for y in range(0, h, major_grid_spacing):
                    cv2.line(self.cached_bg, (0, y), (w, y), major_grid_color, major_thickness)
            
            # Re-bake ALL existing boxes into the new background
            for box in self.boxes:
                self._draw_box_on_image(self.cached_bg, box, vis_scale, current_active_layer)

        # Start from cached background (O(1) copy)
        img = self.cached_bg.copy()
        
        return img

    def _draw_box_on_image(self, img, box, vis_scale, current_layer):
        # Draw rotated boxes using OpenCV
        cx = (box['x'] / self.resolution) * vis_scale
        cy = (box['y'] / self.resolution) * vis_scale
        L = (box['l'] / self.resolution) * vis_scale
        W = (box['w'] / self.resolution) * vis_scale
        theta = box['theta']
        layer = box.get('layer', 0)
        
        # Determine Box Color (Modern Palette)
        if box.get('is_obstacle', False):
             fill_color = (80, 80, 200) # Muted Red
             border_color = (50, 50, 150)
        elif layer == current_layer:
            fill_color = (45, 85, 165) # Warm Brown/Orange (BGR: Blue, Green, Red) -> Actually this is Brownish
            # Previous was (80, 120, 200) -> Blue=80, Green=120, Red=200. This is Orange.
            # Let's use a nice "Cardboard" color or "Active" highlight.
            # "Burnt Sienna": (45, 85, 160)
            fill_color = (45, 85, 160) 
            border_color = (30, 60, 120)
        elif layer == current_layer - 1:
            # Base Layer: Slate Blue / Cool Gray
            fill_color = (168, 147, 122) # Slate Blue (BGR)
            border_color = (100, 80, 60)
        else:
            # Inactive Lower: Light Gray
            fill_color = (190, 190, 190) 
            border_color = (150, 150, 150)
        
        rect = ((cx, cy), (L, W), theta)
        box_pts = cv2.boxPoints(rect)
        box_pts = box_pts.astype(np.int32)
        
        draw_pts = np.zeros_like(box_pts)
        draw_pts[:, 0] = box_pts[:, 1] # Col
        draw_pts[:, 1] = box_pts[:, 0] # Row
        
        # Fill Box
        cv2.drawContours(img, [draw_pts], 0, fill_color, -1, lineType=cv2.LINE_AA)
        
        # Draw Border (Thin, sharp)
        # Use thickness=1 for sharpness on high res, or 2 for visibility
        border_thickness = max(1, int(1 * vis_scale * 0.5)) 
        cv2.drawContours(img, [draw_pts], 0, border_color, border_thickness, lineType=cv2.LINE_AA)
