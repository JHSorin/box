import numpy as np

class MathTools:
    @staticmethod
    def camera_to_robot(point_camera, extrinsic_matrix):
        """
        Convert a point from camera coordinate system to robot coordinate system.
        :param point_camera: [x, y, z] in camera frame
        :param extrinsic_matrix: 4x4 transformation matrix
        :return: [x, y, z] in robot frame
        """
        point_homog = np.append(point_camera, 1)
        point_robot = np.dot(extrinsic_matrix, point_homog)
        return point_robot[:3]

    @staticmethod
    def quaternion_to_euler(w, x, y, z):
        """
        Convert quaternion to euler angles (roll, pitch, yaw)
        """
        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z
