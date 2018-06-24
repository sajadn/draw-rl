import gym
from baselines import deepq
from baselines.common import set_global_seeds
import argparse
from baselines import logger
from graphics import Env
from baselines.common.atari_wrappers import WarpFrame, ScaledFloatFrame,FrameStack


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(3*10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    import time

    current_milli_time = lambda: int(round(time.time() * 1000))

    env = Env(64, 44)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)

    model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (16, 4, 2), (32, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=5e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=100000,
        exploration_fraction=0.05,
        exploration_final_eps=0.01,
        train_freq=2,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        print_freq=30,
        checkpoint_freq=200000,
        prioritized_replay=bool(args.prioritized)
    )
    act.save("draw_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
