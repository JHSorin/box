import time

class ArmInterface:
    def __init__(self):
        self.current_pose = [0, 0, 0, 0, 0, 0]

    def move_j(self, joints):
        """Move joints"""
        print(f"Arm: Moving joints to {joints}")
        time.sleep(0.1) # Simulate delay

    def move_l(self, pose):
        """Move linear to pose [x, y, z, rx, ry, rz]"""
        print(f"Arm: Moving linear to {pose}")
        self.current_pose = pose
        time.sleep(0.1)

    def gripper_on(self):
        print("Arm: Gripper ON")

    def gripper_off(self):
        print("Arm: Gripper OFF")
