#Import necessaries.
from cv2 import VideoCapture, rectangle, putText, imshow, waitKey, FONT_HERSHEY_SIMPLEX, destroyAllWindows
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

def Detect_suspicious(video_path) -> None:
    ''' 
    Detects suspicious objects (e.g., knives) in a video stream using YOLOv8.
    
    Args:
        video_path (str or int): Path to the video file or device index for webcam.
    
    Returns:
        None
    '''
    
    # Open the video capture from the given path
    video = VideoCapture(video_path)

    # Check if the video capture is opened successfully
    if not video.isOpened():
        print("Something went wrong when video capturing!.")
        return

    # Loop to read frames from the video
    while video.isOpened():
        ret, frame = video.read()

        # Break the loop if no frames are returned
        if not ret:
            break

        # Perform detection using the YOLO model
        results = model(frame)

        # Extract detection data
        detections = results[0].boxes.data.cpu().numpy()
            
        # Iterate through the detected objects
        for *box, score, label in detections:
            x1, y1, x2, y2 = map(int, box)
            score = f'{score:.2f}'
            label = model.names[int(label)]

            # Define the label and color based on detection
            if label == 'knife':
                name, color = f'{label} detected!', (0, 0, 255) # Red color for knives
            else:
                name, color = f'{label} {score}', (0, 255, 0) # Green color for the rest
                
            # Draw rectangle and label on the frame
            rectangle(frame, (x1, y1), (x2, y2), color, 2)
            putText(frame, name, (x1, y1 - 10), FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        # Display the frame with detections
        imshow('YOLOv8 Detection', frame)
            
        # Break the loop if 'q' key is pressed
        if waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and destroy all windows
    video.release()
    destroyAllWindows()

# Call the function to detect suspicious objects in a video (0 for webcam)
Detect_suspicious(0)
