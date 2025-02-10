import numpy as np
import pybullet as p
import pybullet_data
import time
from PIL import Image
import os

# --- Configuration Parameters ---
SINGLE_JOINT_STEPS = 10    # Simulation steps per target position for a single joint motion
MULTI_JOINT_STEPS  = 10    # Simulation steps per target position for multiple joints
SLEEP_TIME         = 0.01  # Delay (in seconds) per simulation step

def setup_simulation():
    """
    Initialize the PyBullet simulation, load the robot, and set the camera view.
    """
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF("kuka_iiwa/model.urdf")
    
    # Set camera view with correct parameter names.
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, cameraPitch=-30, 
                                 cameraTargetPosition=[0, 0, 0.5])
    return robot_id

def get_joint_limits(robot_id):
    """
    Retrieve joint limits for each joint. If a joint's limits are identical (i.e. a continuous joint),
    they are set to -π and π.
    """
    num_joints = p.getNumJoints(robot_id)
    joint_limits = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        if lower_limit == upper_limit:  # Continuous joint detected
            lower_limit, upper_limit = -np.pi, np.pi
        joint_limits.append((lower_limit, upper_limit))
    return num_joints, joint_limits

def print_joint_limits(joint_limits):
    """
    Print the limits for each joint.
    """
    for i, (lower, upper) in enumerate(joint_limits):
        print(f"Joint {i}: min = {lower}, max = {upper}")

def reset_robot(robot_id, initial_position=0):
    """
    Reset all joints of the robot to a specified initial position.
    """
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        p.resetJointState(robot_id, i, initial_position)
    p.stepSimulation()

def capture_image(img_folder, img_prefix, identifier):
    """
    Capture an image from the simulation and save it.
    The 'identifier' (string) distinguishes the frame.
    """
    os.makedirs(img_folder, exist_ok=True)
    width, height, img_arr, _, _ = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    # Remove the alpha channel if present
    img = Image.fromarray(img_arr[:, :, :3])
    filename = os.path.join(img_folder, f"{img_prefix}_{identifier}.png")
    img.save(filename)

def create_gif(img_folder, img_prefix, output_file):
    """
    Create a GIF from all images in img_folder whose filenames start with img_prefix.
    """
    images_files = [file for file in sorted(os.listdir(img_folder)) if file.startswith(img_prefix)]
    if images_files:
        images = [Image.open(os.path.join(img_folder, file)) for file in images_files]
        images[0].save(output_file, save_all=True, append_images=images[1:], duration=100, loop=0)

def simulate_continuous_joint_movement(robot_id, joint_index, limits, img_folder, gif_folder, img_prefix):
    """
    For a given joint, move it continuously from its lower to upper limit.
    The robot is reset only once at the beginning.
    """
    os.makedirs(img_folder, exist_ok=True)
    lower, upper = limits

    # Reset robot and set the target joint to its lower limit.
    reset_robot(robot_id)
    p.resetJointState(robot_id, joint_index, lower)
    
    num_positions = 20  # Number of discrete target positions along the motion
    for pos in np.linspace(lower, upper, num=num_positions):
        # Command the joint to move toward the next target position.
        p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=pos)
        # Step the simulation to let the motion occur gradually.
        for _ in range(SINGLE_JOINT_STEPS):
            p.stepSimulation()
            time.sleep(SLEEP_TIME)
        capture_image(img_folder, img_prefix, f"{pos:.2f}")
    
    gif_filename = os.path.join(gif_folder, f"{img_prefix}.gif")
    create_gif(img_folder, img_prefix, gif_filename)

def simulate_continuous_multiple_joints_movement(robot_id, joint_indices, limits_list, img_folder, gif_folder, img_prefix):
    """
    For a group of joints, move them continuously (diagonally) from their lower to upper limits.
    The robot is reset only once at the beginning.
    """
    os.makedirs(img_folder, exist_ok=True)
    reset_robot(robot_id)
    
    # Set each joint in the group to its lower limit.
    for idx, joint_index in enumerate(joint_indices):
        lower, _ = limits_list[idx]
        p.resetJointState(robot_id, joint_index, lower)
    
    num_positions = 20
    # Create a list of target positions for each joint.
    positions_list = [np.linspace(lim[0], lim[1], num=num_positions) for lim in limits_list]
    
    for positions in zip(*positions_list):
        for i, joint_index in enumerate(joint_indices):
            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=positions[i])
        for _ in range(MULTI_JOINT_STEPS):
            p.stepSimulation()
            time.sleep(SLEEP_TIME)
        identifier = "_".join(f"{pos:.2f}" for pos in positions)
        capture_image(img_folder, img_prefix, identifier)
    
    gif_filename = os.path.join(gif_folder, f"{img_prefix}.gif")
    create_gif(img_folder, img_prefix, gif_filename)

def main():
    # Setup simulation and load robot.
    robot_id = setup_simulation()
    num_joints, joint_limits = get_joint_limits(robot_id)
    print(f"The robot has {num_joints} joints with the following limits:")
    print_joint_limits(joint_limits)
    
    # Directories for saving images and GIFs.
    base_folder = "robot_movements"
    img_folder  = os.path.join(base_folder, "images")
    gif_folder  = os.path.join(base_folder, "gifs")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)
    
    # --- Continuous Single Joint Movements ---
    for i in range(num_joints):
        img_prefix = f"joint_{i}_movement"
        print(f"Simulating continuous movement for joint {i}...")
        simulate_continuous_joint_movement(robot_id, i, joint_limits[i], img_folder, gif_folder, img_prefix)
    
    # --- Continuous Multiple Joint Movements ---
    for group_size in range(2, num_joints + 1):
        for start_joint in range(0, num_joints - group_size + 1):
            joint_indices = list(range(start_joint, start_joint + group_size))
            limits_list = [joint_limits[i] for i in joint_indices]
            img_prefix = f"joints_{'_'.join(map(str, joint_indices))}_movement"
            print(f"Simulating continuous movement for joints {joint_indices}...")
            simulate_continuous_multiple_joints_movement(robot_id, joint_indices, limits_list, img_folder, gif_folder, img_prefix)
    
    print("Simulation complete. GIF files and images are saved in the 'robot_movements' folder.")

if __name__ == "__main__":
    main()
