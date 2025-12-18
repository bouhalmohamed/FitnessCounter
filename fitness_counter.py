"""
Fitness Counter - Real-time exercise repetition counter using pose estimation.
Supports squats, push-ups, and bicep curls.
"""

import cv2
import numpy as np
import mediapipe as mp
from enum import Enum


class Exercise(Enum):
    SQUAT = "Squat"
    PUSHUP = "Push-up"
    CURL = "Bicep Curl"


class FitnessCounter:
    """Exercise counter using MediaPipe pose detection."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.exercise = Exercise.SQUAT
        self.count = 0
        self.position = "up"
        self.angle = 0
        
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 150, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points using arctangent."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def get_landmark_coords(self, landmarks, idx, w, h):
        """Convert normalized landmark to pixel coordinates."""
        lm = landmarks.landmark[idx]
        return [int(lm.x * w), int(lm.y * h)]
    
    def process_squat(self, landmarks, w, h):
        """Track squat using knee angle (hip-knee-ankle)."""
        hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
        
        self.angle = self.calculate_angle(hip, knee, ankle)
        
        if self.angle > 160:
            self.position = "up"
        if self.angle < 90 and self.position == "up":
            self.position = "down"
            self.count += 1
        
        return knee
    
    def process_pushup(self, landmarks, w, h):
        """Track push-up using elbow angle (shoulder-elbow-wrist)."""
        shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        
        self.angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.angle > 160:
            self.position = "up"
        if self.angle < 90 and self.position == "up":
            self.position = "down"
            self.count += 1
        
        return elbow
    
    def process_curl(self, landmarks, w, h):
        """Track bicep curl using elbow angle (shoulder-elbow-wrist)."""
        shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        
        self.angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.angle > 150:
            self.position = "down"
        if self.angle < 40 and self.position == "down":
            self.position = "up"
            self.count += 1
        
        return elbow
    
    def draw_ui(self, frame):
        """Draw overlay with counter and exercise info."""
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 80), self.BLACK, -1)
        cv2.rectangle(frame, (0, 78), (w, 82), self.BLUE, -1)
        cv2.putText(frame, "FITNESS COUNTER", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.WHITE, 2)
        cv2.putText(frame, f"Mode: {self.exercise.value}", (w-250, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.GREEN, 2)
        
        # Rep counter
        box_w, box_h = 180, 120
        cv2.rectangle(frame, (20, h-box_h-20), (20+box_w, h-20), self.BLACK, -1)
        cv2.rectangle(frame, (20, h-box_h-20), (23, h-20), self.GREEN, -1)
        cv2.putText(frame, "REPS", (40, h-box_h+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.BLUE, 2)
        cv2.putText(frame, str(self.count), (50, h-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, self.WHITE, 3)
        
        # Angle display
        cv2.rectangle(frame, (w-200, h-100), (w-20, h-20), self.BLACK, -1)
        cv2.putText(frame, f"Angle: {int(self.angle)}", (w-180, h-55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)
        
        color = self.GREEN if self.position == "down" else self.RED
        cv2.circle(frame, (w-110, h-85), 8, color, -1)
        
        # Controls hint
        cv2.putText(frame, "[1] Squat [2] Push-up [3] Curl [R] Reset [Q] Quit",
                   (w//2 - 250, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    
    def run(self):
        """Main application loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access camera")
            return
        
        print("\nFitness Counter")
        print("-" * 30)
        print("[1] Squat  [2] Push-up  [3] Curl")
        print("[R] Reset  [Q] Quit")
        print("-" * 30)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=self.GREEN, thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=self.BLUE, thickness=2)
                )
                
                if self.exercise == Exercise.SQUAT:
                    point = self.process_squat(results.pose_landmarks, w, h)
                elif self.exercise == Exercise.PUSHUP:
                    point = self.process_pushup(results.pose_landmarks, w, h)
                else:
                    point = self.process_curl(results.pose_landmarks, w, h)
                
                cv2.putText(frame, f"{int(self.angle)}", 
                           (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2)
            
            self.draw_ui(frame)
            cv2.imshow("Fitness Counter", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.count = 0
            elif key == ord('1'):
                self.exercise = Exercise.SQUAT
                self.count = 0
            elif key == ord('2'):
                self.exercise = Exercise.PUSHUP
                self.count = 0
            elif key == ord('3'):
                self.exercise = Exercise.CURL
                self.count = 0
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended - {self.exercise.value}s: {self.count}")


if __name__ == "__main__":
    FitnessCounter().run()
