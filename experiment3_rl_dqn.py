import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import gym
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Global Variables
EPISODES = 300
GAMMA = 0.80
EPSILON = 0.65  
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95
BATCH_SIZE = 32
MEMORY_SIZE = 3000
CHECKPOINT_INTERVAL = 10
VIDEO_FOLDER = "outvideos/"
MODEL_FOLDER = "outmodels/"
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'MJPG')
VIDEO_EXT = ".avi"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Robot Environment Class
class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.robot_id = None
        self.joint_index = 0

        # Randomized object position
        self.object_position = [random.uniform(0.8, 1.2), 0, 0.5]

        # Setup the simulation and environment
        self.setup_simulation()
        
        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.video_writer = None

        # Camera parameters for visualization
        self.camera_distance = 2.0
        self.camera_yaw = 90
        self.camera_pitch = -40
        self.camera_target = [0, 0, 0.5]

    def setup_simulation(self):
        # Connect to PyBullet
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.loadURDF("plane.urdf", [0, 0, 0])  
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf")
        self.object_id = p.loadURDF("sphere2.urdf", self.object_position, globalScaling=0.3)
        p.resetJointState(self.robot_id, self.joint_index, 0)
    
    def get_camera_image(self):
        """
        Captures the image from the camera with set parameters.
        """
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2  # Adding the upAxisIndex parameter
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, 
            aspect=1.0, 
            nearVal=0.01, 
            farVal=100.0
        )
        
        width = 640
        height = 480
        img_arr = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)[2]

        # Convert the image from RGBA to BGR format (OpenCV format)
        img_arr = np.array(img_arr)
        img_arr = img_arr[:, :, :3]  # Remove alpha channel
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        return img_arr
        
    def step(self, action):
        p.setJointMotorControl2(self.robot_id, self.joint_index, p.POSITION_CONTROL, targetPosition=action[0])
        p.stepSimulation()
        time.sleep(0.01)

        joint_state = p.getJointState(self.robot_id, self.joint_index)[0]
        distance_to_object = np.linalg.norm(np.array([joint_state, 0, 0.5]) - np.array(self.object_position))

        observation = np.array([float(action[0]), float(joint_state), float(distance_to_object)])

        # Capture camera image for video recording
        if self.video_writer is not None:
            camera_image = self.get_camera_image()
            self.video_writer.write(camera_image)

        reward = -distance_to_object
        done = distance_to_object < 0.1

        return observation, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.setup_simulation()
        joint_state = p.getJointState(self.robot_id, self.joint_index)[0]
        distance_to_object = np.linalg.norm(np.array([joint_state, 0, 0.5]) - np.array(self.object_position))
        return np.array([0.0, joint_state, distance_to_object])

    def close(self):
        p.disconnect()

# Model Creation
def create_model(input_shape, action_space):
    model = Sequential([
        Dense(24, input_dim=input_shape, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    return model

# Train DQN
def train_dqn():
    env = RobotEnv()
    memory = deque(maxlen=MEMORY_SIZE)
    
    input_shape = env.reset().shape[0]
    action_space = 1  # Single continuous action
    model = create_model(input_shape, action_space)
    best_loss = float('inf')

    global EPSILON
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, input_shape])
        episode_loss = 0

        for time_step in range(500):
            action = env.action_space.sample() if np.random.rand() <= EPSILON else model.predict(state, verbose=0)
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])  

            next_state, reward, done, _ = env.step(action)  
            next_state = np.reshape(next_state, [1, input_shape])
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) > BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)

                states = np.vstack([m[0] for m in minibatch])
                actions = np.vstack([m[1] for m in minibatch])
                rewards = np.array([m[2] for m in minibatch])
                next_states = np.vstack([m[3] for m in minibatch])
                dones = np.array([m[4] for m in minibatch])

                q_values = model.predict(states, verbose=0)
                q_next = model.predict(next_states, verbose=0)

                targets = q_values.copy()
                for i in range(BATCH_SIZE):
                    targets[i][0] = rewards[i] if dones[i] else rewards[i] + GAMMA * np.max(q_next[i])

                loss = model.fit(states, targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
                episode_loss += loss.history['loss'][0]  

            if done:
                print(f"\033[92mEpisode: {episode}/{EPISODES}, Score: {time_step}, Epsilon: {EPSILON:.2f}\033[0m")
                break
        
        avg_loss = episode_loss / (time_step + 1) if time_step > 0 else episode_loss

        # Save best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_filename = os.path.join(MODEL_FOLDER, f"checkpoint_episode_{episode}.h5")
            model.save(model_filename)
            checkpoint_video_filename = os.path.join(VIDEO_FOLDER, f"checkpoint_video_episode_{episode}{VIDEO_EXT}")
            env.video_writer = cv2.VideoWriter(checkpoint_video_filename, VIDEO_CODEC, 30, (640, 480))
            print(f"\033[94mBest Loss Improved! New Loss: {avg_loss:.4f} - Model Saved!\033[0m")

        # Save video after every checkpoint
        if episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_video_filename = os.path.join(VIDEO_FOLDER, f"checkpoint_video_episode_{episode}{VIDEO_EXT}")
            env.video_writer = cv2.VideoWriter(checkpoint_video_filename, VIDEO_CODEC, 30, (640, 480))
            print(f"\033[93mCheckpoint: {episode}/{EPISODES} Episodes Completed - Video Saved as {os.path.basename(checkpoint_video_filename)}\033[0m")

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

    # Save final model after training
    model.save(os.path.join(MODEL_FOLDER, "final_model.h5"))
    env.close()

if __name__ == "__main__":
    train_dqn()
