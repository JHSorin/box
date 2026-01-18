import numpy as np
import cv2

class GridPacker:
    def __init__(self, pallet_dims, resolution=20):
        """
        Initialize the Grid Packer.
        
        Args:
            pallet_dims (tuple): (Length_X, Width_Y, Height_Z) in mm.
            resolution (int): Grid cell size in mm. Default 20mm.
        """
        self.pallet_w = pallet_dims[0] # X
        self.pallet_l = pallet_dims[1] # Y
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.grid_w = int(self.pallet_w / self.resolution) # Cols
        self.grid_l = int(self.pallet_l / self.resolution) # Rows
        
        # Initialize grid (0: Free, 1: Occupied)
        self.grid = np.zeros((self.grid_w, self.grid_l), dtype=np.int8)
        
        self.placements = []

    def force_place_box(self, cx, cy, box_dims, theta):
        """
        Force place a box (used for pre-placed obstacles).
        Does not check for collision (assumed empty).
        """
        box_l = box_dims[0]
        box_w = box_dims[1]
        
        self._rasterize_box(cx, cy, box_l, box_w, theta, value=1)
        
        self.placements.append({
            'x': cx, 'y': cy, 'theta': theta, 'forced': True
        })

    def pack(self, box_dims):
        """
        Pack as many boxes as possible using Center-Out Heuristic.
        Supports 0 and 90 degree orientations.
        """
        box_l = box_dims[0]
        box_w = box_dims[1]
        
        # Grid dimensions
        w_grid_0 = int(box_l / self.resolution)
        l_grid_0 = int(box_w / self.resolution)
        
        w_grid_90 = int(box_w / self.resolution)
        l_grid_90 = int(box_l / self.resolution)
        
        while True:
            # Compute Integral Image directly on the grid
            S = self._compute_integral_image(self.grid)
            
            # Find best positions (closest to center)
            cand_0 = self._find_center_candidate(S, w_grid_0, l_grid_0, 0)
            cand_90 = self._find_center_candidate(S, w_grid_90, l_grid_90, 90)
            
            best_candidate = None
            
            if cand_0 and cand_90:
                if cand_0['dist_sq'] < cand_90['dist_sq']:
                    best_candidate = cand_0
                else:
                    best_candidate = cand_90
            elif cand_0:
                best_candidate = cand_0
            elif cand_90:
                best_candidate = cand_90
            
            # Place the best candidate
            if best_candidate:
                self._place_box(best_candidate)
            else:
                break
                
        return self.placements

    def _find_center_candidate(self, S, w_grid, l_grid, theta):
        """
        Find the valid position closest to the center of the pallet.
        """
        max_x = self.grid_w - w_grid
        max_y = self.grid_l - l_grid
        
        if max_x < 0 or max_y < 0:
            return None
            
        # S slicing logic (same as _find_bottom_left_candidate)
        S_tl = S[0 : max_x+1, 0 : max_y+1]
        S_br = S[w_grid : max_x+1+w_grid, l_grid : max_y+1+l_grid]
        S_tr = S[0 : max_x+1, l_grid : max_y+1+l_grid]
        S_bl = S[w_grid : max_x+1+w_grid, 0 : max_y+1]
        
        inner_sum = S_br - S_tr - S_bl + S_tl
        
        # Get all valid top-left indices (x, y)
        valid_indices = np.argwhere(inner_sum == 0)
        
        if valid_indices.size == 0:
            return None
            
        # Calculate distances to center
        cx_grid = self.grid_w / 2.0
        cy_grid = self.grid_l / 2.0
        
        box_cx_offset = w_grid / 2.0
        box_cy_offset = l_grid / 2.0
        
        x_indices = valid_indices[:, 0]
        y_indices = valid_indices[:, 1]
        
        dx = (x_indices + box_cx_offset) - cx_grid
        dy = (y_indices + box_cy_offset) - cy_grid
        
        dist_sq = dx**2 + dy**2
        
        min_idx = np.argmin(dist_sq)
        
        best_x, best_y = valid_indices[min_idx]
        min_dist_sq = dist_sq[min_idx]
        
        cand = {
            'x_idx': best_x,
            'y_idx': best_y,
            'w_grid': w_grid,
            'l_grid': l_grid,
            'theta': theta,
            'dist_sq': min_dist_sq
        }
        
        return cand

    def _compute_integral_image(self, grid_input):
        # grid shape: (W, H)
        # integral shape: (W+1, H+1)
        integ = np.cumsum(np.cumsum(grid_input, axis=0), axis=1)
        S = np.zeros((grid_input.shape[0] + 1, grid_input.shape[1] + 1), dtype=np.int32)
        S[1:, 1:] = integ
        return S

    def _find_bottom_left_candidate(self, S, w_grid, l_grid, theta):
        """
        Find the valid position with minimal Y, then minimal X.
        """
        max_x = self.grid_w - w_grid
        max_y = self.grid_l - l_grid
        
        if max_x < 0 or max_y < 0:
            return None
            
        # Indices for S slicing (Inner Box Check)
        # S coords relative to Grid: TopLeft is (x, y)
        # S[x+w, y+l] - S[x, y+l] - S[x+w, y] + S[x, y]
        
        # S has shape (grid_w+1, grid_l+1)
        # We need results for x in 0..max_x, y in 0..max_y
        # Dimensions of result: (max_x+1, max_y+1)
        
        # S_tl: S[x, y] -> x in 0..max_x, y in 0..max_y
        S_tl = S[0 : max_x+1, 0 : max_y+1]
        
        # S_br: S[x+w, y+l] -> x+w in w..max_x+w, y+l in l..max_y+l
        S_br = S[w_grid : max_x+1+w_grid, l_grid : max_y+1+l_grid]
        
        # S_tr: S[x, y+l]
        S_tr = S[0 : max_x+1, l_grid : max_y+1+l_grid]
        
        # S_bl: S[x+w, y]
        S_bl = S[w_grid : max_x+1+w_grid, 0 : max_y+1]
        
        inner_sum = S_br - S_tr - S_bl + S_tl
        
        # valid_indices returns (x_idx, y_idx) arrays
        # We want minimal Y, then minimal X.
        
        # Transpose inner_sum to (y, x) before argwhere
        valid_indices_T = np.argwhere(inner_sum.T == 0)
        
        if valid_indices_T.size == 0:
            return None
            
        # First element is best (min y, then min x)
        best_y, best_x = valid_indices_T[0]
        
        cand = {
            'x_idx': best_x,
            'y_idx': best_y,
            'w_grid': w_grid,
            'l_grid': l_grid,
            'theta': theta
        }
        
        return cand

    def _place_box(self, cand):
        x_idx = cand['x_idx']
        y_idx = cand['y_idx']
        w_grid = cand['w_grid']
        l_grid = cand['l_grid']
        
        self.grid[x_idx:x_idx+w_grid, y_idx:y_idx+l_grid] = 1
        
        cx = (x_idx * self.resolution) + (w_grid * self.resolution) / 2.0
        cy = (y_idx * self.resolution) + (l_grid * self.resolution) / 2.0
        
        self.placements.append({
            'x': cx,
            'y': cy,
            'theta': cand['theta']
        })

    def _rasterize_box(self, cx, cy, L, W, theta_deg, value=1):
        """
        Rasterize a rotated rectangle onto the grid.
        """
        # Convert physical coords to grid indices (float)
        cx_idx = cx / self.resolution
        cy_idx = cy / self.resolution
        L_idx = L / self.resolution
        W_idx = W / self.resolution
        
        # Get 4 corners of rotated box
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        
        # Half dimensions
        hw = L_idx / 2.0
        hh = W_idx / 2.0
        
        # Local corners [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
        corners_local = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        
        # Rotate
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corners_rot = np.dot(corners_local, rot_matrix.T)
        
        # Translate
        corners_global = corners_rot + np.array([cx_idx, cy_idx])
        
        # Rasterize polygon using OpenCV
        # Grid shape is (W, H). OpenCV expects (H, W) image or (x, y) points.
        # Grid X -> Image Col (x)
        # Grid Y -> Image Row (y)
        # Note: numpy array is [x, y] in our logic (grid_w, grid_l).
        # OpenCV drawContours on a standard image treats it as [row, col].
        # So if we have grid[x, y], this is grid[col, row] in Image terminology?
        # No, grid[row, col] is standard.
        # Let's map Grid X to Row, Grid Y to Col?
        # Usually: Image[y, x].
        # Our grid is defined as [x, y].
        # So grid[x, y] corresponds to Image[x, y].
        # Wait, Image indexing is [row, col].
        # So grid[x, y] means Row=x, Col=y.
        
        # OpenCV points are (x, y) -> (Col, Row).
        # So Point(y, x).
        
        pts = corners_global.astype(np.int32)
        # pts columns are (x_idx, y_idx).
        # If we map x_idx -> Row, y_idx -> Col.
        # Then OpenCV Point should be (Col, Row) = (y, x).
        
        pts_cv = np.zeros_like(pts)
        pts_cv[:, 0] = pts[:, 1] # Col = y
        pts_cv[:, 1] = pts[:, 0] # Row = x
        
        # Draw on a temp grid (uint8)
        # Shape: (Rows, Cols) = (grid_w, grid_l)
        temp_grid = np.zeros((self.grid_w, self.grid_l), dtype=np.uint8)
        
        cv2.fillPoly(temp_grid, [pts_cv], 1)
        
        # Update main grid
        # grid[x, y] = 1 where temp_grid[x, y] == 1
        self.grid[temp_grid == 1] = value
