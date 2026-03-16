from ..Sensor import Sensor
import cv2
import os
import platform
import numpy as np

try:
    import gsdevice
except:
    pass

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img

class GelsightMini(Sensor):
    """
    GelsightMini: A Sensor Class for the GelSight Mini sensor
    """
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.name = "GelSight Mini"
        self.x_offset = 108
        self.y_offset = 110
        self.z_offset = 67
        self.z_clearance = 2
        self.max_penetration = 3.5
        self.default_calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.csv")

    if platform.system() == "Linux":
        def connect(self):
            """
            Connects to the GelSight Mini sensor.
            """
            # Code to connect to the sensor
            self.sensor = gsdevice.Camera("GelSight Mini")
            self.sensor.connect()
            
        def disconnect(self):
            """
            Disconnects from the GelSight Mini sensor.
            """
            # Code to disconnect from the sensor
            self.sensor.stop_video()

        def capture_image(self):
            """
            Captures an image from the GelSight Mini sensor.
            """
            # Code to return an image from the sensor
            image = cv2.cvtColor(self.sensor.get_image(), cv2.COLOR_BGR2RGB)
            return cv2.flip(image, 1)
        
    elif platform.system() == "Windows" or platform.system() == "Darwin":
        def connect(self):
            """
            Connects to the GelSight Mini sensor.
            """
            # Code to connect to the sensor
            self.sensor = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

            if self.sensor is None or not self.sensor.isOpened():
                print('Warning: unable to open video source...')
            
        def disconnect(self):
            """
            Disconnects from the GelSight Mini sensor.
            """
            # Code to disconnect from the sensor
            self.sensor.release()

        def capture_image(self):
            """
            Captures an image from the GelSight Mini sensor.
            """
            # Code to return an image from the sensor
            _, image = self.sensor.read()
            
            image = resize_crop_mini(image, 320, 240)

            # return cv2.flip(image, 1)
            return image