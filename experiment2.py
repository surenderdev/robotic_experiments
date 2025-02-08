import numpy as np
import pybullet as p
import pybullet_data
import time
from PIL import Image
import os

def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF("kuka_iiwa/model.urdf")
    
    cameraDistance = 2.5
    cameraYaw = 90
    cameraPitch = -30
    targetPosition = [0, 0, 0.5]
    
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, targetPosition)
    
    return robot_id

def get_joint_limits(robot_id):
    num_joints = p.getNumJoints(robot_id)
    joint_limits = [(p.getJointInfo(robot_id, i)[8], p.getJointInfo(robot_id, i)[9]) for i in range(num_joints)]
    return num_joints, joint_limits

def print_joint_limits(joint_limits):
    for i, limits in enumerate(joint_limits):
        print(f"Joint {i}: min = {limits[0]}, max = {limits[1]}")

def move_joint(robot_id, joint_index, target_position):
    p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
    p.stepSimulation()
    time.sleep(0.1)

def capture_image(img_folder, img_prefix, target_position):
    width, height, img, _, _ = p.getCameraImage(640, 480)
    img = Image.fromarray(img)
    img.save(os.path.join(img_folder, f"{img_prefix}_{target_position}.png"))

def create_gif(img_folder, img_prefix, output_file):
    images = [Image.open(os.path.join(img_folder, file)) for file in sorted(os.listdir(img_folder)) if file.startswith(img_prefix)]
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=100, loop=0)

def simulate_joint_movement(robot_id, joint_index, limits, img_folder, gif_folder, img_prefix):
    os.makedirs(img_folder, exist_ok=True)
    for position in np.linspace(limits[0], limits[1], num=20):
        move_joint(robot_id, joint_index, position)
        capture_image(img_folder, img_prefix, position)
    create_gif(img_folder, img_prefix, os.path.join(gif_folder, f"{img_prefix}.gif"))

def move_multiple_joints_and_capture(robot_id, joint_indices, target_positions, img_folder, img_prefix):
    for idx, joint_index in enumerate(joint_indices):
        p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_positions[idx])
    p.stepSimulation()
    time.sleep(0.1)
    width, height, img, _, _ = p.getCameraImage(640, 480)
    img = Image.fromarray(img)
    img.save(os.path.join(img_folder, f"{img_prefix}_{'_'.join(map(str, target_positions))}.png"))

def simulate_multiple_joints_movement(robot_id, joint_indices, limits_list, img_folder, gif_folder, img_prefix):
    os.makedirs(img_folder, exist_ok=True)
    num_positions = 20
    positions_list = [np.linspace(limits[0], limits[1], num=num_positions) for limits in limits_list]
    for positions in zip(*positions_list):
        move_multiple_joints_and_capture(robot_id, joint_indices, positions, img_folder, img_prefix)
    create_gif(img_folder, img_prefix, os.path.join(gif_folder, f"{img_prefix}.gif"))

def main():
    robot_id = setup_simulation()
    num_joints, joint_limits = get_joint_limits(robot_id)
    print(f"The robot has {num_joints} joints with the following limits:")
    print_joint_limits(joint_limits)

    base_folder = "robot_movements"
    img_folder = os.path.join(base_folder, "images")
    gif_folder = os.path.join(base_folder, "gifs")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)

    for i in reversed(range(num_joints)):
        img_prefix = f"joint_{i}_movement"
        simulate_joint_movement(robot_id, i, joint_limits[i], img_folder, gif_folder, img_prefix)

    for num_joints_to_move in range(2, num_joints + 1):
        for start_joint in range(0, num_joints - num_joints_to_move + 1):
            joint_indices = list(range(start_joint, start_joint + num_joints_to_move))
            limits_list = [joint_limits[i] for i in joint_indices]
            img_prefix = f"joints_{'_'.join(map(str, joint_indices))}_movement"
            simulate_multiple_joints_movement(robot_id, joint_indices, limits_list, img_folder, gif_folder, img_prefix)

    print("GIF files and images created for each joint and multiple joint movements.")

if __name__ == "__main__":
    main()
