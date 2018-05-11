from baselines import deepq
from graphics import Env
import argparse
from baselines import logger
import numpy as np
from baselines.common.atari_wrappers import WarpFrame, ScaledFloatFrame, FrameStack
from time import sleep
def main():
    env = Env(64, 64)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, 1)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (32, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.25,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        restore = True
    )
    for _ in range(100):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            sleep(0.01)
            env.render()
            action = act(np.array(obs)[None])[0]
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
            # print(action, rew)
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()
