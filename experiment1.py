#Author: Surender Varma
#
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
from ultralytics import YOLOWorld

# -------------------------------
# Setup YOLO and PyBullet Environments
# -------------------------------

confthr=0.05
iouthr=0.4

definedclasses=["robot", "bench", "ball"]
#definedclasses=["robot"]
model = YOLOWorld('yolov8x-worldv2.pt')
model.set_classes(definedclasses)


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Loading scene along with objects (initial scene setup)
planeId = p.loadURDF("plane.urdf")
ballId1 = p.loadURDF("sphere2.urdf", basePosition=[0.5, 0, 0.2], globalScaling=1.0,useFixedBase=True)
ballId2 = p.loadURDF("sphere2.urdf", basePosition=[-0.5, -2, 1], globalScaling=1.5)
p.changeVisualShape(ballId2, linkIndex=-1, rgbaColor=[1, 0, 0, 0.6])

benchId = p.loadURDF("table/table.urdf", basePosition=[1, 2, 0.1], globalScaling=1.5)

robot_start = [1.5, -2.5, 0.3]  # Adjusted position
robotId = p.loadURDF("r2d2.urdf", basePosition=robot_start, globalScaling=2.2, useFixedBase=True)

# Object to move
sourceObject = ballId1

# -------------------------------
# Video Writer Setup
# -------------------------------
WIDTH = 640
HEIGHT = 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('robot_animation.mp4', fourcc, 20.0, (WIDTH, HEIGHT))

# -------------------------------
# Camera Capture
# -------------------------------
def capture_frame():
    """Capture a frame from PyBullet and return an 8-bit BGR image."""
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(WIDTH, HEIGHT)
    # rgbImg is a flat list; reshape to (height, width, 4)
    rgb_array = np.reshape(rgbImg, (height, WIDTH, 4)).astype(np.uint8)
    frame = cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)
    return frame

# -------------------------------
# Object detection and localisation
# -------------------------------
def detect_objects(confthr, iouthr):
    """Detect objects in the saved 'current_scene.jpg' and convert the detection to 3D coordinates."""
    results = model.predict('current_scene.jpg',conf=confthr, iou=iouthr)
    detections = results[0].boxes

    print("detections", detections)
    detected_objects = []
    
    # Reload the current scene image to annotate
    frame = cv2.imread('current_scene.jpg')
    
    for detection in detections:
        xyxy = detection.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
        label = model.names[int(detection.cls[0].item())]
        print("label:", label)
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        
        detected_objects.append({
            "bbox": xyxy,
            "label": label,
            "center": (x_center, y_center)
        })
                
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                      (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('detected_objects.jpg', frame)
    video_out.write(frame)
    return detected_objects


# -------------------------------
# Ball Movement and Placement
# -------------------------------
def move_object(sourceobjID, target_xy, target_z):
    """
    Move the ball from its current position to target_xy (x,y)
    and lower it to target_z.
    """
    initial_position, _ = p.getBasePositionAndOrientation(sourceobjID)
    print("Ball initial position:", initial_position)
    # Compute a lift height to avoid collision with the bench
    lift_height = max(initial_position[2] + 0.05, target_z + 0.1)
    print("Lift height set to:", lift_height)
    
    # Phase 1: Lift the ball vertically.
    for h in np.linspace(initial_position[2], lift_height, num=50):
        p.resetBasePositionAndOrientation(sourceobjID, (initial_position[0], initial_position[1], h), [0, 0, 0, 1])
        p.stepSimulation()
        frame = capture_frame()
        video_out.write(frame)
        time.sleep(0.01)
    
    # Phase 2: Move horizontally toward the target.
    for t in np.linspace(0, 1, num=100):
        new_x = (1 - t) * initial_position[0] + t * target_xy[0]
        new_y = (1 - t) * initial_position[1] + t * target_xy[1]
        p.resetBasePositionAndOrientation(sourceobjID, (new_x, new_y, lift_height), [0, 0, 0, 1])
        p.stepSimulation()
        frame = capture_frame()
        video_out.write(frame)
        time.sleep(0.01)
    
    # Phase 3: Lower the ball onto the target (bench) surface.
    for h in np.linspace(lift_height, target_z, num=100):
        p.resetBasePositionAndOrientation(sourceobjID, (target_xy[0], target_xy[1], h), [0, 0, 0, 1])
        p.stepSimulation()
        frame = capture_frame()
        video_out.write(frame)
        time.sleep(0.01)
    
    # Phase 4: Allow settling
    for _ in range(500):
        p.stepSimulation()
        frame = capture_frame()
        video_out.write(frame)
        time.sleep(0.005)
    
    print("Ball movement complete.")

# Capture and save the initial scene
initial_frame = capture_frame()
cv2.imwrite('current_scene.jpg', initial_frame)

# Run detection
perception_data = detect_objects(confthr,iouthr)
print("Detected Objects:", perception_data)

# -------------------------------
# Video Writer Setup
# -------------------------------
# Set resolution. Make sure these match the camera parameters below.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('robot_animation.mp4', fourcc, 20.0, (WIDTH, HEIGHT))
video_out.write(initial_frame)

# -------------------------------
# Determine the Target for Placement
# -------------------------------
# We expect the (bench or table) to be our target. Sometimes the label may differ.
target_obj = None
if perception_data:
    for obj in perception_data:
        print("obj:", obj)
        if "bench" in obj["label"].lower() or "table" in obj["label"].lower():
            target_obj = obj
            print("success")
            break

# Fallback: if detection yields an unexpected position, use the bench's simulation pose.
if target_obj:
    print("Using detected target:", target_obj)
    # The detected 3D position may not reflect the actual table top height.
    # Adjust the target z by comparing with the bench's known position.
    detected_target = target_obj["center"]
    bench_pos, _ = p.getBasePositionAndOrientation(benchId)
    print("Detected_target:", detected_target)
    print("Actual:", bench_pos[0], bench_pos[1])
    # For our URDF, the bench’s top may be bench_pos[2] + ~1.2 (adjust as needed).
    expected_top_z = bench_pos[2] + 1.2
    # Use the detected center coordinates directly
    target_xy = detected_target
    target_xy = (bench_pos[0], bench_pos[1])
    target_z = expected_top_z + 0.02
else:
    print("No valid bench/table detected. Falling back to simulation data.")
    bench_pos, _ = p.getBasePositionAndOrientation(benchId)
    target_xy = (bench_pos[0], bench_pos[1])
    target_z = bench_pos[2] + 1.2

print("Placing ball at target:", target_xy, target_z)

move_object(sourceObject, target_xy, target_z)


# -------------------------------
# Extra Frames for Stability
# -------------------------------
for _ in range(100):
    p.stepSimulation()
    video_out.write(capture_frame())
    time.sleep(0.05)

video_out.release()
p.disconnect()
print("Video saved as 'robot_animation.mp4'.")
