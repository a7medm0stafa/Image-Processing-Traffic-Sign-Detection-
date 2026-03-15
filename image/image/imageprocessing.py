import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class SignDetectionNode(Node):
    def __init__(self):
        super().__init__('sign_detection_node')
        self.bridge = CvBridge()
        
        # Initialize Camera (0 = laptop cam, change to 1 or 2 if using external USB)
        self.cap = cv2.VideoCapture(0)
        
        # Publishers for Milestone requirements [cite: 18, 21, 39]
        self.command_publisher = self.create_publisher(String, 'vehicle/command', 10)
        
        # Timer: Calls processing every 0.05s (approx 20 FPS)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info("MCTR1010: Vision Pipeline and Camera Started")

    def timer_callback(self):
        start_time = time.time()
        ret, frame = self.cap.read()
        
        if ret:
            # 1. Geometric: Crop Right Side (Requirement 3a) [cite: 31]
            roi = self.crop_right_side(frame)
            
            # 2. Geometric: Resize/Scale (Requirement 3a) [cite: 31]
            scaled = self.resize_scale(roi)
            
            # 3. Intensity: Adjust Brightness (Requirement 3b) [cite: 32]
            bright = self.adjust_brightness(scaled, value=30)
            
            # 4. Intensity: Contrast Enhancement (Requirement 3b) [cite: 32]
            contrast = self.enhance_contrast(bright)
            
            # 5. Smoothing: Gaussian Blur (Requirement 3c) [cite: 33]
            blurred = self.apply_gaussian_blur(contrast)
            
            # 6. Feature Extraction: HSV Conversion (Requirement 34) [cite: 34]
            hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # Decision Logic (Requirement 4: Closed Loop) 
            command = self.make_decision(hsv_img)
            self.command_publisher.publish(String(data=command))

            # Display results for Video Deliverables [cite: 73, 74]
            cv2.imshow("1. Original Feed", frame)
            cv2.imshow("2. Pipeline Output", hsv_img)
            cv2.waitKey(1)

            # Performance Data for Report 
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000
            self.get_logger().info(f"Command: {command} | Processing: {elapsed:.2f}ms")
#geometric 1
    def crop_right_side(self, image):
        h, w, _ = image.shape
        return image[0:h, w//2:w]
#geometric 2
    def resize_scale(self, image):
        return cv2.resize(image, (320, 240))
#intensity 1
    def adjust_brightness(self, image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
#intensity 2
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
#smoothing
    def apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def make_decision(self, hsv_image): 
        avg_saturation = np.mean(hsv_image[:,:,1])
        
        # Feature 1: Check for Red saturation (Traffic Light)
        if avg_saturation > 150:
            return f"STOP, avg saturation {avg_saturation:.2f}"
        
        return f"GO, avg saturation {avg_saturation:.2f}"

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()