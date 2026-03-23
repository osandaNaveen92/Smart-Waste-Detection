from ultralytics import YOLO
import cv2
import json
import datetime
import os

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\Acer\Documents\ET\smart_waste_detection\runs\detect\outputs\garbage_v15\weights\best.pt"
EVIDENCE_DIR = "evidence"
OUTPUT_DIR = "outputs/events"
CAMERA_ID = "CAM_01"
ZONE = "Zone_A"
CONFIDENCE_THRESHOLD = 0.5

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LITTERING DETECTOR — links person to garbage
# ============================================================
class LitteringDetector:
    def __init__(self, proximity_thresh=120, frames_to_confirm=30):
        self.proximity_thresh = proximity_thresh
        self.frames_to_confirm = frames_to_confirm
        self.tracked_garbage = {}
        self.person_last_seen = {}
        self.confirmed_events = []
        self.reported_ids = set()

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def distance(self, c1, c2):
        import math
        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def update(self, frame_id, persons, garbage_objects):
        current_person_ids = set()

        for person in persons:
            pid = person['id']
            current_person_ids.add(pid)
            self.person_last_seen[pid] = frame_id

        for obj in garbage_objects:
            oid = obj['id']
            obj_center = self.get_center(obj['box'])

            if oid not in self.tracked_garbage:
                for person in persons:
                    p_center = self.get_center(person['box'])
                    if self.distance(obj_center, p_center) < self.proximity_thresh:
                        self.tracked_garbage[oid] = {
                            'first_frame': frame_id,
                            'linked_person_id': person['id'],
                            'linked_person_box': person['box'],
                            'obj_class': obj['class'],
                            'obj_box': obj['box'],
                            'confirmed': False
                        }
                        break
            else:
                entry = self.tracked_garbage[oid]
                frames_since = frame_id - entry['first_frame']
                linked_pid = entry['linked_person_id']

                if (frames_since >= self.frames_to_confirm
                        and linked_pid not in current_person_ids
                        and not entry['confirmed']
                        and oid not in self.reported_ids):
                    entry['confirmed'] = True
                    self.reported_ids.add(oid)
                    self.confirmed_events.append({
                        'frame_id': frame_id,
                        'person_id': linked_pid,
                        'person_box': entry['linked_person_box'],
                        'garbage_class': entry['obj_class'],
                        'garbage_box': entry['obj_box']
                    })

        return self.confirmed_events


# ============================================================
# SAVE EVIDENCE IMAGE
# ============================================================
def save_evidence(frame, person_box, event):
    x1, y1, x2, y2 = [int(c) for c in person_box]
    pad = 30
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)

    crop = frame[y1:y2, x1:x2]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"littering_{timestamp}_p{event['person_id']}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(filepath, crop)
    return filepath


# ============================================================
# SAVE JSON EVENT LOG
# ============================================================
def save_event_log(event_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"event_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(event_data, f, indent=2)
    print(f"\n📝 Event logged: {filepath}")
    return filepath


# ============================================================
# DRAW BOXES ON FRAME
# ============================================================
def draw_detections(frame, persons, garbage_objects, events):
    # Draw persons in blue
    for p in persons:
        x1, y1, x2, y2 = [int(c) for c in p['box']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, f"Person {p['id']}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # Draw garbage in green
    for obj in garbage_objects:
        x1, y1, x2, y2 = [int(c) for c in obj['box']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{obj['class']} {obj['conf']:.0%}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw littering alert in red
    for event in events:
        x1, y1, x2, y2 = [int(c) for c in event['garbage_box']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "⚠ LITTERING!", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Overlay info
    cv2.putText(frame, f"Zone: {ZONE} | Cam: {CAMERA_ID}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================
if __name__ == '__main__':

    # Load models
    print("Loading models...")
    garbage_model = YOLO(MODEL_PATH)         # your trained garbage model
    person_model = YOLO("yolov8s.pt")        # pretrained COCO model for person detection
    detector = LitteringDetector(proximity_thresh=120, frames_to_confirm=30)
    print("✅ Models loaded!")

    # ---- INPUT SOURCE ----
    # For webcam:         source = 0
    # For video file:     source = r"C:\path\to\video.mp4"
    # For RTSP stream:    source = "rtsp://username:password@ip:port/stream"
    source = 0  # ⬅️ change this to your video file or RTSP stream

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Could not open video source")
        exit()

    print(f"✅ Video source opened: {source}")
    print("Press 'Q' to quit\n")

    frame_id = 0
    GARBAGE_CLASSES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or stream lost.")
            break

        # --- Detect garbage ---
        g_results = garbage_model.track(
            frame, persist=True, conf=CONFIDENCE_THRESHOLD,
            device='cpu', verbose=False
        )

        # --- Detect persons ---
        p_results = person_model.track(
            frame, persist=True, conf=0.5,
            classes=[0],   # class 0 = person in COCO
            device='cpu', verbose=False
        )

        persons = []
        garbage_objects = []

        # Parse person detections
        if p_results[0].boxes.id is not None:
            for box in p_results[0].boxes:
                persons.append({
                    'id': int(box.id),
                    'box': box.xyxy[0].tolist()
                })

        # Parse garbage detections
        if g_results[0].boxes.id is not None:
            for box in g_results[0].boxes:
                cls_idx = int(box.cls)
                garbage_objects.append({
                    'id': int(box.id),
                    'box': box.xyxy[0].tolist(),
                    'class': GARBAGE_CLASSES[cls_idx],
                    'conf': float(box.conf)
                })

        # --- Run littering detection ---
        events = detector.update(frame_id, persons, garbage_objects)

        # --- Handle new confirmed littering events ---
        for event in events:
            evidence_path = save_evidence(frame, event['person_box'], event)

            log = {
                "event_id": f"EVT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_P{event['person_id']}",
                "timestamp": datetime.datetime.now().isoformat(),
                "camera_id": CAMERA_ID,
                "zone": ZONE,
                "person_id": event['person_id'],
                "garbage_type": event['garbage_class'],
                "confidence": 0.85,
                "evidence_image": evidence_path,
                "frame_id": frame_id
            }

            print(json.dumps(log, indent=2))
            save_event_log(log)

        # --- Draw and show ---
        frame = draw_detections(frame, persons, garbage_objects, events)
        cv2.imshow("Smart Waste Detection", frame)

        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detection session ended.")
    print(f"📁 Evidence saved in: {EVIDENCE_DIR}")
    print(f"📁 Event logs saved in: {OUTPUT_DIR}")