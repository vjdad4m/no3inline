import numpy as np
import torch


def gen_rollouts_for_counterex(num_rollouts, grid_size, counterex):
    row_idx, column_idx = np.where(counterex == 1)
    point_locs = list(zip(row_idx, column_idx))

    rollouts = []

    for k in range(num_rollouts):
        states = [counterex, ]
        np.random.seed(k)
        np.random.shuffle(point_locs)

        for i in range(len(point_locs)):
            prev_state = states[-1].copy()
            prev_state[point_locs[i][0], point_locs[i][1]] = 0
            states.append(prev_state)

        rollouts.append(torch.tensor(np.array(states[::-1])).view(-1, grid_size * grid_size))

    return list(zip(rollouts, [0] * len(rollouts)))


def main():
    grid_size = 6
    num_rollouts = 100

    counterex = np.loadtxt(f"test/counterex_{grid_size}x{grid_size}.txt", delimiter=',', dtype=float)

    rollouts = gen_rollouts_for_counterex(num_rollouts, grid_size, counterex)

    print(len(rollouts))

    print(rollouts[0])


if __name__ == "__main__":
    main()
