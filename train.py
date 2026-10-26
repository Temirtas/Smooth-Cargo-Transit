from stable_baselines3 import PPO
from cargo_env import CargoDelivery3DEnv

# 1. Instantiate the environment
env = CargoDelivery3DEnv(render_mode="direct")  # "direct" for fast headless training

# 2. Train PPO Agent
print("Training PPO Agent in PyBullet 3D Simulation...")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=30000)

# Save model weights
model.save("ppo_cargo_robot")
print("Training Complete! Saved as 'ppo_cargo_robot.zip'")

# 3. Render Trained Model Live
env_visual = CargoDelivery3DEnv(render_mode="human")
obs, _ = env_visual.reset()

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env_visual.step(action)
    if terminated or truncated:
        obs, _ = env_visual.reset()

env_visual.close()