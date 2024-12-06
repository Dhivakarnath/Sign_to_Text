import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"F:\objDet\best.pt")  # Path to your best.pt

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Real-time detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO detection on the current frame
    results = model.predict(source=frame, conf=0.4, save=False, save_txt=False)  # Run detection

    # Extract detections for the current frame
    detections = results[0].boxes.data  # Get bounding boxes and scores
    if len(detections) > 0:
        for detection in detections:
            # Extract bounding box coordinates, confidence, and class index
            x1, y1, x2, y2, conf, cls = detection[:6]
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(cls)  # Convert class to integer

            # Get the class name from the model's names
            class_name = results[0].names[cls]
            
            # Format the label as "Class: Confidence"
            label = f"{class_name}: {conf:.2f}"

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the bounding box and label
    cv2.imshow("YOLO Live Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
