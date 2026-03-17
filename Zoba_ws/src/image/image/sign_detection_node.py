import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import time
import os


class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # ROS Publisher
        self.command_publisher = self.create_publisher(
            String,
            'vehicle/command',
            10
        )

        # Run pipeline ~20 FPS
        self.timer = self.create_timer(0.05, self.timer_callback)

        # Setup logging and image saving
        self.frame_count = 0
        self.log_file = open("ros_processing_log.txt", "a")
        os.makedirs("ros_output_images", exist_ok=True)
        self.log_file.write("--- STARTED NEW ROS SESSION ---\n")

        self.get_logger().info("Vision pipeline started")


    def timer_callback(self):

        start_time = time.time()

        ret, frame = self.cap.read()

        if not ret:
            return

        # -------------------------
        # Preprocessing Pipeline
        # -------------------------

        t0 = time.time()
        roi = self.crop_right_side(frame)
        t1 = time.time()

        scaled = self.resize_scale(roi)
        t2 = time.time()

        bright = self.adjust_brightness(scaled, 30)
        t3 = time.time()

        contrast = self.enhance_contrast(bright)
        t4 = time.time()

        blurred = self.apply_gaussian_blur(contrast)
        t5 = time.time()

        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        t6 = time.time()

        # -------------------------
        # Traffic Light Detection
        # -------------------------

        command, masks = self.detect_traffic_light(hsv_img)

        # Publish command
        self.command_publisher.publish(String(data=command))

        # -------------------------
        # Debug Windows & Save
        # -------------------------

        red_mask, yellow_mask, green_mask = masks

        # Save ALL steps for the very first frame to show the pipeline stages
        if self.frame_count == 0:
            cv2.imwrite("ros_output_images/00_original.jpg", frame)
            cv2.imwrite("ros_output_images/01_cropped.jpg", roi)
            cv2.imwrite("ros_output_images/02_resized.jpg", scaled)
            cv2.imwrite("ros_output_images/03_brightened.jpg", bright)
            cv2.imwrite("ros_output_images/04_contrasted.jpg", contrast)
            cv2.imwrite("ros_output_images/05_blurred.jpg", blurred)
            cv2.imwrite("ros_output_images/06_hsv.jpg", hsv_img)
            cv2.imwrite("ros_output_images/07_red_mask.jpg", red_mask)
            cv2.imwrite("ros_output_images/08_yellow_mask.jpg", yellow_mask)
            cv2.imwrite("ros_output_images/09_green_mask.jpg", green_mask)

        cv2.imshow("Original Camera", frame)
        cv2.imshow("ROI", roi)
        cv2.imshow("Processed (HSV)", hsv_img)

        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("Green Mask", green_mask)

        cv2.waitKey(1)

        # -------------------------
        # Performance log
        # -------------------------
        
        # Log to file and print to terminal
        log_str = (f"Frame {self.frame_count:04d} -> "
                   f"Crop: {(t1 - t0) * 1000:.2f}ms | "
                   f"Resize: {(t2 - t1) * 1000:.2f}ms | "
                   f"Bright: {(t3 - t2) * 1000:.2f}ms | "
                   f"Contrast: {(t4 - t3) * 1000:.2f}ms | "
                   f"Blur: {(t5 - t4) * 1000:.2f}ms | "
                   f"HSV Conv: {(t6 - t5) * 1000:.2f}ms | "
                   f"Total Preprocess: {(t6 - t0) * 1000:.2f}ms\n")
        
        print(log_str, end="")
        self.log_file.write(log_str)
        self.log_file.flush()

        end_time = time.time()
        elapsed = (end_time - start_time) * 1000

        self.get_logger().info(f"Command: {command} | Processing time: {elapsed:.2f} ms")
        self.frame_count += 1


    # --------------------------------
    # Geometric Transformation 1
    # Crop Right Side
    # --------------------------------

    def crop_right_side(self, image):

        h, w, _ = image.shape

        return image[:, w//2:w]


    # --------------------------------
    # Geometric Transformation 2
    # Resize / Scaling
    # --------------------------------

    def resize_scale(self, image):

        return cv2.resize(image, (320, 240))


    # --------------------------------
    # Intensity Transformation
    # Brightness Adjustment
    # --------------------------------

    def adjust_brightness(self, image, value):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        v = cv2.add(v, value)

        v = np.clip(v, 0, 255)

        hsv = cv2.merge((h, s, v))

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # --------------------------------
    # Contrast Enhancement (CLAHE)
    # --------------------------------

    def enhance_contrast(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8,8)
        )

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


    # --------------------------------
    # Image Smoothing
    # Gaussian Blur
    # --------------------------------

    def apply_gaussian_blur(self, image):

        return cv2.GaussianBlur(image, (5,5), 0)


    # --------------------------------
    # Traffic Light Detection
    # --------------------------------

    def detect_traffic_light(self, hsv):

        # RED ranges (two ranges in HSV)
        lower_red1 = np.array([0,120,70])
        upper_red1 = np.array([10,255,255])

        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([179,255,255])

        # GREEN (moved higher to avoid yellow)
        lower_green = np.array([50,100,100])
        upper_green = np.array([85,255,255])

        # YELLOW (expanded range - much wider)
        lower_yellow = np.array([10,50,50])
        upper_yellow = np.array([40,255,255])


        # Create masks

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = red_mask1 + red_mask2

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


        # Count detected pixels

        red_pixels = np.sum(red_mask)

        yellow_pixels = np.sum(yellow_mask)

        green_pixels = np.sum(green_mask)


        threshold = 5000


        # Decision logic

        if red_pixels > threshold:

            command = "STOP"

        elif yellow_pixels > threshold:

            command = "SLOW"

        elif green_pixels > threshold:

            command = "GO"

        else:

            command = "NO_SIGNAL"


        return command, (red_mask, yellow_mask, green_mask)


# --------------------------------
# Main
# --------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = SignDetectionNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    node.cap.release()
    node.log_file.close()

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
