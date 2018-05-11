from graphics import Env
import time


def main():
    env = Env(64, 64)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            try:
                action = int(input())
            except ValueError:
                continue
            obs, rew, done, dict = env.step(action)
            episode_rew += rew
            time.sleep(0.1)
            print(action, rew)
            print("Reward", episode_rew)
            print("coverage", dict['coverage'])

if __name__ == '__main__':
    main()
