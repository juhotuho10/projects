from stable_baselines3 import PPO

from Pong_play_env import Pong_play_env

RENDER = True

TIMESTEPS = 5_000
version = 1

env = Pong_play_env()
env.reset()

model_folder = "1702215195"

model_path = f"Pong-{version}/{model_folder}/best_model.zip"

model = PPO.load(model_path, env=env)

# Enjoy trained agent
obs, _ = env.reset()

done_count = 0

p1_score = 0
p2_score = 0

# while loop has to be here since for loop would only render x amount of steps
# but a while loop checks how many times the environment is completed
while done_count <= 100:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)

    # print(obs)

    if RENDER:
        env.render(p1_score, p2_score)

    if done:
        p1_score += info["p1_won"]
        p2_score += info["p2_won"]

        obs, _ = env.reset()
        done_count += 1
