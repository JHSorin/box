import cv2
import numpy as np

class Viewer2D:
    def __init__(self, resolution=20.0, box_dims=(200, 200)):
        self.resolution = resolution
        self.box_dims = box_dims # (L, W)
        self.window_name = "Palletizing System Monitor (2D)"
        self.auto_mode = False
        self.next_clicked = False
        
        # Initialize OpenCV Window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 800)

    def update(self, rgb_image, target_pose=None):
        """
        Update the view using OpenCV HighGUI (Lightweight).
        """
        if rgb_image is None:
            return

        display_img = rgb_image.copy()
        
        # 1. Flip vertically to match grid coordinates (Y-up)
        # VirtualPallet image (H, W) -> Row=X, Col=Y?
        # Simulation:
        # cx_idx = cx / res (X axis)
        # cy_idx = cy / res (Y axis)
        # corners_global = [cx_idx, cy_idx]
        # pts[:, 0] = pts[:, 1] (Col=y)
        # pts[:, 1] = pts[:, 0] (Row=x)
        # So Image: Row=X, Col=Y.
        
        # If we want standard view where X is horizontal and Y is vertical:
        # We need Row=Y, Col=X.
        # So we need to transpose: (Col=X, Row=Y)
        display_img = cv2.transpose(display_img)
        
        # Now Image: Row=Y, Col=X.
        # But OpenCV/Image origin is Top-Left (Y=0 at top).
        # Physical world Y=0 is usually "bottom".
        # So we flip vertically (Flip around X axis).
        display_img = cv2.flip(display_img, 0) 
        
        # Now Image:
        # Row 0 (Top) -> High Y
        # Row H (Bottom) -> Low Y
        # Col 0 (Left) -> Low X
        # Col W (Right) -> High X
        # This matches standard 2D plot.
        
        # Draw Target Pose if available
        if target_pose:
             # Calculate dynamic vis_scale based on image width
             # Image width = grid_w * vis_scale
             # We assume image width corresponds to full pallet width?
             # Yes, get_rgb_image returns full pallet view.
             
             # But we don't know grid_w here.
             # However, we can use the ratio of Image Width (px) / Pallet Width (mm).
             # Pixels per mm.
             
             img_h, img_w = display_img.shape[:2]
             
             # Pallet Width in mm. We need this info.
             # Ideally passed in init, but default resolution is 20.
             # If resolution is 1, pallet is 2000.
             # Let's try to deduce scale if we assume standard pallet.
             # Or better: Just use resolution if we trust it matches the image generation.
             
             # In Simulation:
             # w_px = (W_grid) * vis_scale = (W_mm / res) * vis_scale
             # So vis_scale = w_px * res / W_mm.
             
             # If we don't know W_mm, we can't get vis_scale exactly.
             # BUT:
             # x_px = (x_mm / res) * vis_scale
             #      = x_mm * (vis_scale / res)
             #      = x_mm * (w_px * res / W_mm / res)
             #      = x_mm * (w_px / W_mm)
             
             # So we just need pixels_per_mm = ImageWidth / PalletWidth.
             # We can assume Pallet Width = 2000mm (standard large) or read from config.
             # Let's read from config in main and pass it, or just default to 2000mm if not provided.
             # Or better: simulation.py uses self.grid_w.
             
             # Workaround: Assume 2000mm width for now as seen in config.
             pallet_w_mm = 2000.0
             pixels_per_mm = img_w / pallet_w_mm
             
             # Target Pose in mm
             tx_mm = target_pose['x']
             ty_mm = target_pose['y']
             theta = target_pose['theta']
             
             # Convert to Image Pixels
             x_px = int(tx_mm * pixels_per_mm)
             
             # y_px (from bottom)
             y_px_bottom = int(ty_mm * pixels_per_mm)
             y_px = img_h - y_px_bottom
             
             # Draw Center
             # Use Magenta for Target Center to stand out against new palette
             cv2.drawMarker(display_img, (x_px, y_px), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
             
             # Draw Box Outline
             # Use configured box dimensions
             box_l = self.box_dims[0]
             box_w = self.box_dims[1]
             
             # Box in image coords (pixels)
             box_l_px = box_l * pixels_per_mm
             box_w_px = box_w * pixels_per_mm
             
             rect = ((x_px, y_px), (box_l_px, box_w_px), -theta) 
             
             box_pts = cv2.boxPoints(rect)
             box_pts = np.int32(box_pts)
             # Draw thick Magenta outline for target box
             cv2.drawContours(display_img, [box_pts], 0, (255, 0, 255), 3)

        # Add UI Text Overlay - Cleaned up
        # status_text = "AUTO" if self.auto_mode else "PAUSED"
        # color = (0, 255, 0) if self.auto_mode else (0, 165, 255)
        # cv2.putText(display_img, f"Mode: {status_text} (A)", (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.putText(display_img, "Next: Space/N", (10, 60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Only show text if paused or critical info?
        # User requested clean view. Let's move status to window title or minimal.
        # Window title update is cleaner.
        status_str = "AUTO" if self.auto_mode else "PAUSED - Press Space/N for Next"
        cv2.setWindowTitle(self.window_name, f"Palletizing Monitor (2D) - {status_str}")

        cv2.imshow(self.window_name, display_img)

    def wait_for_input(self):
        """
        Handle OpenCV window events.
        """
        self.next_clicked = False
        
        while not self.next_clicked:
            # Wait for key press (1ms delay)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or key == 27: # Esc or q
                return # Exit?
            elif key == ord(' ') or key == ord('n'):
                self.next_clicked = True
            elif key == ord('a'):
                self.auto_mode = not self.auto_mode
                # Update UI immediately to show status change
                # We can't easily redraw without the image, but next loop will update.
                print(f"UI: Auto Mode set to {self.auto_mode}")
                if self.auto_mode:
                    break # Exit wait loop if switched to auto

            # If Auto Mode, break immediately (but add small delay to see frames)
            if self.auto_mode:
                cv2.waitKey(10) # Small delay for visualization
                break
                
            # Check if window closed
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    def close(self):
        cv2.destroyAllWindows()
