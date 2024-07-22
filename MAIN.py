import cv2
import numpy as np
from shapely.geometry import Polygon, box

# Load YOLO model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

classes = open('coco.names').read().strip().split('\n')

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')

# Define parking space polygon vertices (example coordinates)
parking_polygon = np.array([[272,130], [332,253], [420,254], [362,129]], np.int32)  # Polygon vertices

# Scaling factor for bounding box
scaling_factor = 0.3  # Increase bounding box size by 30%

def is_inside(polygon, box):
    poly = Polygon(polygon)
    bx, by, bw, bh = box
    box_polygon = Polygon([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)])
    return poly.contains(box_polygon)

def touches_polygon(polygon, box):
    poly = Polygon(polygon)
    bx, by, bw, bh = box
    box_polygon = Polygon([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)])
    return poly.intersects(box_polygon) and not poly.contains(box_polygon)

def is_outside(polygon, box):
    poly = Polygon(polygon)
    bx, by, bw, bh = box
    box_polygon = Polygon([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)])
    return not poly.intersects(box_polygon)

frame_skip = 1  # Number of frames to skip

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

# Skip frames
    for _ in range(frame_skip):
        cap.grab()

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to 'car' in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Apply scaling factor
                w = int(w * scaling_factor)
                h = int(h * scaling_factor)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Ensure the bounding box is within the frame
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (0, 255, 0)  # Default color for correctly parked car
            status_text = "CP" # Correctly Parked

            # Check if the bounding box is inside, touches, or is outside the parking polygon
            if is_outside(parking_polygon, (x, y, w, h)):
                color = (0, 0, 255)  # Red for incorrectly parked car
                status_text = "ICP" # Incorrectly Parked
            elif touches_polygon(parking_polygon, (x, y, w, h)):
                color = (0, 255, 255)  # Yellow for touching the polygon
                status_text = "ICP" # Incorrectly Parked
            elif not is_inside(parking_polygon, (x, y, w, h)):
                color = (0, 0, 255)  # Red for incorrectly parked car
                status_text = "ICP" # Incorrectly Parked
            
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Display the status text
            cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Draw the parking polygon
    cv2.polylines(frame, [parking_polygon], isClosed=True, color=(255, 0, 0), thickness=3)  # Blue polygon for parking space

    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
