import cv2
import numpy as np
import time
import serial
import glob
import os

def find_arduino_port():
    # specifically check for the user's exact arduino string
    import os
    exact_match = '/dev/cu.usbmodem11201'
    exact_match_space = '/dev/cu.usbmodem 11201' # sometimes users type with space
    
    if os.path.exists(exact_match):
        return exact_match
    if os.path.exists(exact_match_space):
        return exact_match_space
        
    ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/cu.usbserial*')
    if ports:
        return ports[0]
    return None

class TrafficSignDetector:
    def __init__(self, serial_port=None):
        # Initialize camera (0 is usually the built-in webcam)
        self.cap = cv2.VideoCapture(0)
        
        # Setup logging and image saving
        self.frame_count = 0
        self.log_file = open("local_processing_log.txt", "a")
        os.makedirs("local_output_images", exist_ok=True)
        self.log_file.write("--- STARTED NEW SESSION ---\n")
        
        # Try to connect to Arduino
        self.serial_port = None
        if serial_port is None:
            serial_port = find_arduino_port()
            
        if serial_port:
            try:
                self.serial_port = serial.Serial(serial_port, 9600, timeout=1)
                print(f"Connected to Arduino on {serial_port}")
                time.sleep(2) # Wait for Arduino to reset after serial connection
            except Exception as e:
                print(f"Failed to connect to Arduino on {serial_port}: {e}")
                self.serial_port = None
        else:
            print("WARNING: Arduino not found. Running in vision-only mode.")

    # --------------------------------
    # Geometric Transformations
    # --------------------------------
    def crop_right_side(self, image):
        h, w, _ = image.shape
        return image[:, w//2:w]

    def resize_scale(self, image):
        return cv2.resize(image, (320, 240))

    # --------------------------------
    # Intensity Transformations
    # --------------------------------
    def adjust_brightness(self, image, value):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # --------------------------------
    # Image Smoothing
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

        # GREEN
        lower_green = np.array([35,100,100])
        upper_green = np.array([85,255,255])

        # YELLOW
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([30,255,255])

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

    def send_command_to_arduino(self, command):
        if not self.serial_port:
            return
            
        try:
            if command == "STOP":
                self.serial_port.write(b'R\n')
            elif command == "GO":
                self.serial_port.write(b'G\n')
            elif command == "SLOW":
                self.serial_port.write(b'Y\n')
            else:
                self.serial_port.write(b'O\n')
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")

    def run(self):
        print("Starting local vision pipeline and actuator control... Press 'q' to quit.")
        
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera. Exiting.")
                break

            # Preprocessing Pipeline
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

            # Log to terminal and file
            log_str = (f"Frame {self.frame_count:04d} -> "
                       f"Crop: {(t1 - t0) * 1000:.2f}ms | "
                       f"Resize: {(t2 - t1) * 1000:.2f}ms | "
                       f"Bright: {(t3 - t2) * 1000:.2f}ms | "
                       f"Contrast: {(t4 - t3) * 1000:.2f}ms | "
                       f"Blur: {(t5 - t4) * 1000:.2f}ms | "
                       f"HSV Conv: {(t6 - t5) * 1000:.2f}ms | "
                       f"Total Preprocess: {(t6 - t0) * 1000:.2f}ms\n")
            
            print(log_str, end="") # Print to terminal
            self.log_file.write(log_str) # Write to log file
            self.log_file.flush()

            # Traffic Light Detection
            command, masks = self.detect_traffic_light(hsv_img)
            red_mask, yellow_mask, green_mask = masks

            # Save ALL steps for the very first frame to show the pipeline stages
            if self.frame_count == 0:
                cv2.imwrite("local_output_images/00_original.jpg", frame)
                cv2.imwrite("local_output_images/01_cropped.jpg", roi)
                cv2.imwrite("local_output_images/02_resized.jpg", scaled)
                cv2.imwrite("local_output_images/03_brightened.jpg", bright)
                cv2.imwrite("local_output_images/04_contrasted.jpg", contrast)
                cv2.imwrite("local_output_images/05_blurred.jpg", blurred)
                cv2.imwrite("local_output_images/06_hsv.jpg", hsv_img)
                cv2.imwrite("local_output_images/07_red_mask.jpg", red_mask)
                cv2.imwrite("local_output_images/08_yellow_mask.jpg", yellow_mask)
                cv2.imwrite("local_output_images/09_green_mask.jpg", green_mask)

            self.frame_count += 1

            # Debug Windows
            cv2.imshow("Original Camera", frame)
            cv2.imshow("ROI", roi)
            cv2.imshow("Processed (HSV)", hsv_img)
            cv2.imshow("Red Mask", red_mask)
            cv2.imshow("Yellow Mask", yellow_mask)
            cv2.imshow("Green Mask", green_mask)

            # Log Performance and Status
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000
            print(f"Command: {command} | Processing time: {elapsed:.2f} ms")

            # Send Command to Arduino
            self.send_command_to_arduino(command)

            # Wait briefly and check for quit key (q)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.log_file.close()
        if self.serial_port:
            self.serial_port.close()
            print("Serial connection closed.")

if __name__ == '__main__':
    detector = TrafficSignDetector()
    detector.run()
