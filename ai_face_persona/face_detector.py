"""
face_detector.py
MediaPipe Face Mesh based face detector + landmarks.
Returns bounding box and list of landmarks in pixel coordinates.

Build by llzipperl
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    import mediapipe as mp
except Exception:
    mp = None

_HAS_MP_SOLUTIONS = bool(mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'))
mp_face_mesh = mp.solutions.face_mesh if _HAS_MP_SOLUTIONS else None


class FaceDetector:
    """Wraps MediaPipe Face Mesh for detection and landmarks extraction.

    Methods
    -------
    detect(frame)
        Returns bbox (x,y,w,h) and landmarks list [(x,y), ...] in pixel coords.
    """

    def __init__(self, refine_landmarks: bool = True, max_faces: int = 1):
        self.max_faces = max_faces
        self.face_mesh = None
        self.haar = None
        self.use_mediapipe = _HAS_MP_SOLUTIONS

        if self.use_mediapipe:
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                                   max_num_faces=max_faces,
                                                   refine_landmarks=refine_landmarks,
                                                   min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5)
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int,int,int,int]], List[Tuple[int,int]]]:
        """Detect faces and landmarks in BGR frame.

        Returns
        -------
        bbox: (x, y, w, h) or None
        landmarks: list of (x, y) in pixel coords (empty if none)
        """
        if frame is None:
            return None, []

        h, w = frame.shape[:2]

        if self.use_mediapipe and self.face_mesh is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.face_mesh.process(img_rgb)
            except Exception:
                return None, []

            if not results.multi_face_landmarks:
                return None, []

            # Use first face only
            face_lms = results.multi_face_landmarks[0]

            landmarks = []
            x_vals = []
            y_vals = []
            for lm in face_lms.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                landmarks.append((px, py))
                x_vals.append(px)
                y_vals.append(py)

            if not x_vals or not y_vals:
                return None, landmarks

            x_min = max(min(x_vals) - 10, 0)
            y_min = max(min(y_vals) - 10, 0)
            x_max = min(max(x_vals) + 10, w)
            y_max = min(max(y_vals) + 10, h)

            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            return bbox, landmarks

        # Fallback: Haar face detection without landmarks.
        if self.haar is None or self.haar.empty():
            return None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None, []

        x, y, ww, hh = max(faces, key=lambda b: b[2] * b[3])
        return (int(x), int(y), int(ww), int(hh)), []


if __name__ == "__main__":
    # Quick smoke-test to ensure import runs without error
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox, lms = detector.detect(frame)
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow('fd', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
