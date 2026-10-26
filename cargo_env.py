import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class CargoDelivery3DEnv(gym.Env):
    """
    3D Physics Simulation of a Wheeled Mobile Robot delivering cargo.
    Goal: Drive to target quickly while minimizing high acceleration spikes.
    """
    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        
        # Action space: [steering_angle (-0.5 to 0.5 rad), throttle_velocity (-10.0 to 20.0 rad/s)]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -5.0], dtype=np.float32),
            high=np.array([0.5, 20.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [x, y, dx_to_goal, dy_to_goal, linear_vel, angular_vel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        self.goal_pos = np.array([4.0, 4.0, 0.0])
        self.prev_velocity = 0.0

        # Racecar URDF joint indices
        self.steering_joints = [0, 2]     # Front wheel steering hinges
        self.motor_joints = [3, 5, 7]      # Driving wheels (rear & gear)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not p.isConnected(self.physics_client):
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        
        # Load 3D Plane and 3D Robot URDF Model
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.05], physicsClientId=self.physics_client)
        
        # Load 3D Visual Goal Marker
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[1, 0, 0, 1], physicsClientId=self.physics_client)
        p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=self.goal_pos, physicsClientId=self.physics_client)
        
        self.prev_velocity = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        
        dx = self.goal_pos[0] - pos[0]
        dy = self.goal_pos[1] - pos[1]
        
        speed = np.linalg.norm(vel[:2])
        
        return np.array([pos[0], pos[1], dx, dy, speed, ang_vel[2]], dtype=np.float32)

    def step(self, action):
        if not p.isConnected(self.physics_client):
            raise RuntimeError("PyBullet GUI window was closed.")

        self.step_count += 1
        steering, throttle = action[0], action[1]
        
        # Apply steering to front wheel hinges
        for wheel in self.steering_joints:
            p.setJointMotorControl2(
                self.robot_id, wheel, p.POSITION_CONTROL,
                targetPosition=steering, force=10,
                physicsClientId=self.physics_client
            )
            
        # Apply throttle velocity to driving wheels
        for wheel in self.motor_joints:
            p.setJointMotorControl2(
                self.robot_id, wheel, p.VELOCITY_CONTROL,
                targetVelocity=throttle, force=10,
                physicsClientId=self.physics_client
            )
        
        p.stepSimulation(physicsClientId=self.physics_client)
        if self.render_mode == "human":
            time.sleep(1.0 / 60.0)
            
        obs = self._get_obs()
        dist_to_goal = np.linalg.norm(obs[2:4])
        current_speed = obs[4]
        
        # Calculate acceleration (derivative of speed)
        accel = abs(current_speed - self.prev_velocity) / 0.05
        self.prev_velocity = current_speed
        
        # Reward Function: Distance reduction minus acceleration penalty
        reward = -dist_to_goal - (0.1 * accel)
        
        terminated = False
        if dist_to_goal < 0.5:
            reward += 100.0
            terminated = True
            
        truncated = self.step_count >= 300
        return obs, reward, terminated, truncated, {}

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)

if __name__ == "__main__":
    env = CargoDelivery3DEnv(render_mode="human")
    obs, _ = env.reset()
    
    print("Testing 3D PyBullet Environment with Motor Control...")
    try:
        for _ in range(300):
            # Sample continuous action [steering, throttle]
            action = np.array([0.2, 15.0], dtype=np.float32)  # Turn slightly right & drive forward
            obs, reward, done, truncated, _ = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
    except RuntimeError as e:
        print(e)
    finally:
        env.close()