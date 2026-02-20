import cv2 
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
model = YOLO("yolov10m.pt")
tracker = DeepSort(max_age=20,
        n_init=2,
        nms_max_overlap=0.3,
        max_cosine_distance=0.8,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    ) 
cap=cv2.VideoCapture("cars2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
detections = []
names = model.names
# print(names)
line_y = 360
counted_ids = set()
track_history = {}
car_count = 0
while True :
    ret,frame=cap.read()
    if not ret :
        break
    results = model(frame, classes=[1,2,3,5,6,7])[0]
    detections = []
    for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls=int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], cls,conf))
    tracks = tracker.update_tracks(detections, frame=frame)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id=track.track_id
        l,t,r,b=map(int,track.to_ltrb()) 
        cx = (l + r) // 2
        cy = (t + b) // 2
        if cy > line_y and track_id not in counted_ids:
            counted_ids.add(track_id)
            car_count += 1
 
        cv2.rectangle(frame,(l,t),(r,b),(0,0,255),2)

        cv2.putText(frame,f"{track_id} {names[cls]}",(l, t - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),2)
        cv2.putText(
    frame,
    f"Cars Count: {car_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 255),
    2
)

    cv2.imshow("tracking",frame)
    if cv2.waitKey(delay) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()