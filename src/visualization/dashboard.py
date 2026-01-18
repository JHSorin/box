import cv2
import numpy as np

class Dashboard:
    def __init__(self):
        pass

    def update(self, rgb_image, status_text):
        """
        Overlay status on the image.
        """
        if rgb_image is None:
            return

        # Make a copy to draw on
        display_img = rgb_image.copy()
        
        # Draw text
        cv2.putText(display_img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # In a headless environment, we might save the image instead of showing it
        # cv2.imshow("Dashboard", display_img)
        # cv2.waitKey(1)
        
        print(f"[Dashboard] Status: {status_text}")
