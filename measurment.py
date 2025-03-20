import cv2 
import mediapipe as mp
import numpy as np
from collections import deque

class BodyMeasurement:
    def __init__(self, smoothing_frames=10):  
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.smoothing_frames = smoothing_frames
        self.body_size_buffer = deque(maxlen=smoothing_frames)
        self.last_stable_size = None  


        def classify_body_size(self, hip_width_px):
            """Classify body size based on pixel width."""
            if hip_width_px < 3062-3251:
                return "XS (Extra Small)" 
            elif hip_width_px < 3289-3478:
                return "S (Small)"
            elif hip_width_px < 3516-3706:
                return "M (Medium)"
            elif hip_width_px < 3743-3932:
                return "L (Large)"
            elif hip_width_px < 3970-4159:
                return "XL (Extra Large)"
            elif hip_width_px < 4197-4386:
                return "XXL (Double Extra Large)"
            elif hip_width_px < 4424-4613:
                return "XXXL+ (Extra Large)"
            else:
                return "ERROR"


        def classify_body_size(self, weist_width_px):
            """Classify body size based on pixel width."""
            if weist_width_px < 2495-2684:
                return "XS (Extra Small)" 
            elif weist_width_px < 2722-2912:
                return "S (Small)"
            elif weist_width_px < 2950-3139:
                return "M (Medium)"
            elif weist_width_px < 3177-3366:
                return "L (Large)"
            elif weist_width_px < 3404-3593:
                return "XL (Extra Large)"
            elif weist_width_px < 3631-3821:
                return "XXL (Double Extra Large)"
            elif weist_width_px < 3858-4048:
                return "XXXL+ (Extra Large)"
            else:
                return "ERROR"

        def classify_body_size(self, shoulder_width_px):
            """Classify body size based on pixel width."""
            if shoulder_width_px < 1436-1550:
                return "XS (Extra Small)" 
            elif shoulder_width_px < 1588-1701:
                return "S (Small)"
            elif shoulder_width_px <1739-1853:
                return "M (Medium)"
            elif shoulder_width_px < 1890-2004:
                return "L (Large)"
            elif shoulder_width_px < 2042-2155:
                return "XL (Extra Large)"
            elif shoulder_width_px < 2193-2307:
                return "XXL (Double Extra Large)"
            elif shoulder_width_px < 2344-2458:
                return "XXXL+ (Extra Large)"
            else:
                return "ERROR"
    def classify_body_size(self, chest_width_px):
        """Classify body size based on pixel width."""
        if chest_width_px < 1436:
            return "XS (Extra Small)"
        elif chest_width_px < 1588:
            return "S (Small)"
        elif chest_width_px < 1739:
            return "M (Medium)"
        elif chest_width_px < 1890:
            return "L (Large)"
        elif chest_width_px < 2042:
            return "XL (Extra Large)"
        elif chest_width_px < 2193:
            return "XXL (Double Extra Large)"
        elif  chest_width_px < 2344:
            return "XXXL+ (Extra Large)"
        else:
            return "ERROR"
    
    def draw_grid(self, image):
        """Draws a 3x3 grid on the image."""
        height, width, _ = image.shape
        step_x, step_y = width // 3, height // 3
        color = (50, 50, 50)  # Darker color for better visibility
        thickness = 2  
        
        for i in range(1, 3):
            cv2.line(image, (i * step_x, 0), (i * step_x, height), color, thickness)
            cv2.line(image, (0, i * step_y), (width, i * step_y), color, thickness)

    def draw_position_box(self, image):
        """Draws a bounding box for proper standing position."""
        height, width, _ = image.shape
        box_x1, box_x2 = int(0.3 * width), int(0.7 * width)  # 30% to 70% width
        box_y1, box_y2 = int(0.2 * height), int(0.8 * height)  # 20% to 80% height

        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
        cv2.putText(image, "Stand inside the box", (box_x1, box_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return (box_x1, box_x2, box_y1, box_y2)

    def estimate_measurements(self, image):
        """Estimate body size and check for proper standing position."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        results = self.pose.process(image_rgb)
        body_size = "Not in Frame"

        # Get frame guide position
        box_x1, box_x2, box_y1, box_y2 = self.draw_position_box(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Ensure both shoulders are detected
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                cv2.putText(image, "Make sure your shoulders are visible!", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return image, "Not in Frame"

            # Compute shoulder width (normalize by frame size)
            shoulder_x1 = left_shoulder.x * width
            shoulder_x2 = right_shoulder.x * width
            shoulder_width_px = abs(shoulder_x2 - shoulder_x1)

            # Check if shoulders are inside the bounding box
            if shoulder_x1 < box_x1 or shoulder_x2 > box_x2:
                cv2.putText(image, "Move left or right to center yourself", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return image, "Not Centered"
            
            MIN_SHOULDER_WIDTH=1436 #XS (Smallest standard size)
            MAX_SHOULDER_WIDTH=2458 #XXXL+(Largest standard size)

            if shoulder_width_px < MIN_SHOULDER_WIDTH:
                cv2.putText(image, "Move closer to the camera", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return image, "Too Far"

            if shoulder_width_px > MAX_SHOULDER_WIDTH:
                cv2.putText(image, "Move farther from the camera", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return image, "Too Close"

            # Store in buffer for smoothing
            self.body_size_buffer.append(shoulder_width_px)
            avg_shoulder_width_px = np.mean(self.body_size_buffer)

            # Classify body size
            new_body_size = self.classify_body_size(avg_shoulder_width_px)

            # Stabilization: Confirm the size if it remains consistent
            if len(self.body_size_buffer) == self.smoothing_frames:
                stable_avg = np.mean(self.body_size_buffer)
                stable_size = self.classify_body_size(stable_avg)
                self.last_stable_size = stable_size
            
            
        # Display stable size if available
        if self.last_stable_size:
            body_size = self.last_stable_size
            cv2.putText(image, f"Body Size: {body_size}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        
        # Draw grid overlay
        self.draw_grid(image)
        
        return image, body_size


def main():
    cap = cv2.VideoCapture(0)
    body_measurement = BodyMeasurement(smoothing_frames=10)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        processed_frame, body_size = body_measurement.estimate_measurements(frame)
        cv2.imshow('Body Size Classification', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
