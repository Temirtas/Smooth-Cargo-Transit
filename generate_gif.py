import imageio
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from cargo_env import CargoDelivery3DEnv

def capture_simulation_gif(model_path="ppo_cargo_robot.zip", output_gif="demo.gif", num_steps=150):
    env = CargoDelivery3DEnv(render_mode="direct")
    
    # Try loading trained model, fallback to test action if not trained yet
    try:
        model = PPO.load(model_path)
        use_model = True
        print("Loaded trained PPO model.")
    except Exception:
        use_model = False
        print("No trained model found. Using default driving actions...")

    frames = []
    obs, _ = env.reset()
    
    width, height = 480, 480
    print("Capturing 3D frames for GIF generation...")
    
    for _ in range(num_steps):
        if use_model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.2, 15.0], dtype=np.float32)  # Test driving action
            
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Center camera on robot
        robot_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.physics_client)
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=robot_pos,
            distance=3.5,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=env.physics_client
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0,
            physicsClientId=env.physics_client
        )
        
        img_data = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=env.physics_client
        )
        
        # Extract RGB array, reshape correctly, and cast to uint8
        rgb_buffer = img_data[2]
        rgb_array = np.reshape(rgb_buffer, (height, width, 4))[:, :, :3]
        rgb_array = np.uint8(rgb_array)
        
        frames.append(rgb_array)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    
    # Save frames as GIF
    imageio.mimsave(output_gif, frames, fps=20)
    print(f"GIF created successfully as '{output_gif}'!")

if __name__ == "__main__":
    capture_simulation_gif()