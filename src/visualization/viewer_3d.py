import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import yaml

class Viewer3D:
    def __init__(self, config_path='config/pallet_config.yaml', box_config_path='config/box_config.yaml'):
        # Load Configs
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.resolution = config.get('grid_resolution', 1.0)
        
        with open(box_config_path, 'r') as f:
            box_config = yaml.safe_load(f)
        self.box_dims = box_config['dimensions'] # [L, W, H]
        
        # Adjust layout to make room for buttons at bottom
        # Use single large plot for clearer visualization
        self.fig = plt.figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15) 
        
        self.fig.suptitle("Palletizing System Monitor")
        
        # UI State
        self.auto_mode = False
        self.next_clicked = False
        
        # Buttons
        # [left, bottom, width, height]
        ax_auto = plt.axes([0.35, 0.02, 0.15, 0.05])
        ax_next = plt.axes([0.55, 0.02, 0.15, 0.05])
        
        self.btn_auto = Button(ax_auto, 'Auto/Pause (A)')
        self.btn_auto.on_clicked(self.toggle_auto)
        
        self.btn_next = Button(ax_next, 'Next Step (Spc)')
        self.btn_next.on_clicked(self.next_step)
        
        # Keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.ion() # Interactive mode

    def on_key(self, event):
        if event.key == ' ' or event.key == 'n':
            self.next_step(None)
        elif event.key == 'a':
            self.toggle_auto(None)
            
    def next_step(self, event):
        self.next_clicked = True
        
    def toggle_auto(self, event):
        self.auto_mode = not self.auto_mode
        print(f"UI: Auto Mode set to {self.auto_mode}")

    def wait_for_input(self):
        """
        Blocks execution until user presses Next or if in Auto mode.
        """
        if self.auto_mode:
            plt.pause(0.1)
            return
            
        print("UI: Waiting for user input... (Press 'Next' or Spacebar)")
        self.next_clicked = False
        while not self.next_clicked and not self.auto_mode:
            plt.pause(0.1)
            if not plt.fignum_exists(self.fig.number):
                print("UI: Window closed.")
                break

    def update(self, height_map, rgb_image=None, target_pose=None):
        """
        Update the view using Matplotlib.
        """
        # Optimize: Don't clear axes completely to avoid layout recalculation overhead
        # self.ax1.clear()
        
        # Display ONLY the Semantic RGB View (strictly following 4-color rule)
        if rgb_image is not None:
            # Convert BGR (OpenCV) to RGB (Matplotlib)
            rgb_vis = rgb_image[:, :, ::-1] # BGR to RGB
            
            # Transpose to match height map orientation (Row=Y, Col=X)
            rgb_vis = np.transpose(rgb_vis, (1, 0, 2))
            
            # Use extent to map high-res image back to grid coordinates (0..W, 0..H)
            # Assuming scale factor is 4 as per Simulation
            h, w, _ = rgb_vis.shape
            grid_w = w / 4.0
            grid_h = h / 4.0
            
            if not hasattr(self, 'img_handle'):
                self.img_handle = self.ax1.imshow(rgb_vis, origin='lower', extent=[0, grid_w, 0, grid_h])
                self.ax1.set_title("Plan View (High-Res 2D)")
                self.ax1.set_xlabel("X (Grid)")
                self.ax1.set_ylabel("Y (Grid)")
            else:
                self.img_handle.set_data(rgb_vis)
                self.img_handle.set_extent([0, grid_w, 0, grid_h])
        else:
            # Fallback if no RGB image
            self.ax1.clear()
            self.ax1.imshow(height_map.T, cmap='gray', origin='lower')
            self.ax1.set_title("Height Map (Grayscale)")

        # Handle Target Pose Drawing
        # Clear previous target markers
        for artist in self.ax1.lines + self.ax1.patches:
             artist.remove()
        
        if target_pose:
            # Draw target box
            # Grid coordinates
            tx = target_pose['x'] / self.resolution
            ty = target_pose['y'] / self.resolution
            theta = target_pose['theta']
            
            self.ax1.plot(tx, ty, 'r+', markersize=15, markeredgewidth=3, label='Target Center')
            
            # Draw rotated rectangle
            box_l_grid = self.box_dims[0] / self.resolution
            box_w_grid = self.box_dims[1] / self.resolution
            
            # Rectangle centered at (0,0) first
            rect = patches.Rectangle((-box_l_grid/2, -box_w_grid/2), box_l_grid, box_w_grid, 
                                     linewidth=2, edgecolor='r', facecolor='none')
            
            # Apply transformation: Rotate -> Translate
            t = transforms.Affine2D().rotate_deg(theta).translate(tx, ty) + self.ax1.transData
            rect.set_transform(t)
            
            self.ax1.add_patch(rect)
            # self.ax1.legend() # Skip legend re-creation for speed

        plt.draw()
        plt.pause(0.01) # Short pause to update UI
        
        # Optimize: Don't save frame every single update to avoid massive I/O lag
        # Only save if explicitly requested or at low frequency?
        # For now, let's remove it or safeguard it.
        # try:
        #    self.fig.savefig('data/logs/latest_view.png')
        # except:
        #    pass
            
    def close(self):
        plt.close(self.fig)
