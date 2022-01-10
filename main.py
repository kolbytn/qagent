from env import QEnv


if __name__ == "__main__":
    env = QEnv()
    obs = env.reset()
    env.render()

    done = False
    while not done:
        action = int(input())
        obs, reward, done, info = env.step(action)
        env.render()