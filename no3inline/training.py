from functools import cmp_to_key
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import visualize
from config import HYPERPARAMETERS
import random

import no3inline
import no3inline.models
import wandb


def calculate_rewards_per_state(state_list, N, reward_type):
    if reward_type == 'summed':
        reward = 0
        for state in state_list:
            reward += no3inline.calculate_reward(state.view(N, N))
    elif reward_type == 'laststate':
        state = state_list[-1]
        reward = no3inline.calculate_reward(state.view(N, N))
    return reward

def generate_batched_rollout(model, N, BATCH_SIZE, device, reward_type):
    states = [torch.zeros((BATCH_SIZE, N * N)).float()]

    for _ in range(2 * N):
        new_state = states[-1].clone()
        
        # Select random acation with probability EPSILON
        if torch.rand(1) > HYPERPARAMETERS['EPSILON']:
            next_state_probabilities = model(states[-1].reshape((BATCH_SIZE, 1, N, N)).to(device)).cpu().detach()
            next_state_probabilities = torch.softmax(next_state_probabilities, dim=1)
            
            action = torch.multinomial(next_state_probabilities, num_samples=1).squeeze()
        else:
            empty_cells = torch.where(new_state == 0)
            action = [random.choice(empty_cells[1][empty_cells[0] == i]) for i in range(BATCH_SIZE)]

        new_state[torch.arange(BATCH_SIZE), action] = 1
        
        states.append(new_state)

    rollouts = torch.stack(states).reshape((2 * N + 1, BATCH_SIZE, N * N))
    rollouts = rollouts.permute((1, 0, 2))

    rewards = [calculate_rewards_per_state(rollout, N, reward_type) for rollout in rollouts]
    
    return list(zip(rollouts, rewards))

def tensor_from_rollout(rollout, N):
    stack = rollout.view(len(rollout), N, N)
    tensor_list = torch.zeros((2, len(rollout) - 1, N, N))
    tensor_list[0] = stack[:-1]
    tensor_list[1] = stack[1:]
    return tensor_list

def get_training_data(top_k, N, device):
    return [
        tensor_from_rollout(rollout[0], N).to(device)
        for rollout in top_k
    ]

def train_epoch(data, model, criterion, optimizer, N):
    epoch_loss = []
    for X, y in data:
        optimizer.zero_grad()
        output = model(X.view((2 * N, 1, N, N)))
        loss = criterion(output, torch.where((y != X).flatten(1))[1])
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)

def train(HYPERPARAMETERS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device = }')

    if HYPERPARAMETERS['MODEL'] == 'convnet':
        inner_model = no3inline.models.ConvNet(HYPERPARAMETERS['N'])
    elif HYPERPARAMETERS['MODEL'] == 'resenet':
        inner_model = no3inline.models.ResNet18()
    else: # default
        inner_model = no3inline.models.ConvNet(HYPERPARAMETERS['N'])

    model = no3inline.models.MaskedLogitNetwork(inner_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()

    print(f'Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    rollouts = []
    previous_best_reward = HYPERPARAMETERS['N'] * HYPERPARAMETERS['N'] * 2
    tq = tqdm.trange(HYPERPARAMETERS['N_EPOCHS'])
    
    def reward_sort(pair1, pair2):
        if pair1[1] < pair2[1]:
            return -1
        if pair1[1] > pair2[1]:
            return 1
        if str(pair1[0][-1]) < str(pair2[0][-1]):
            return -1
        return 1

    for i in tq:
        rollouts.extend(generate_batched_rollout(model, HYPERPARAMETERS['N'], HYPERPARAMETERS['N_ROLLOUTS'], device, HYPERPARAMETERS['REWARD_TYPE']))
    
        rollouts.sort(key = cmp_to_key(reward_sort))

        if HYPERPARAMETERS['DEDUPLICATION']:
            alt_rollouts = []
            alt_rollouts.append(rollouts[0])
            for index in range(1, len(rollouts)):
                if not torch.all(torch.eq(rollouts[index-1][0][-1], rollouts[index][0][-1])):
                    alt_rollouts.append(rollouts[index])
            rollouts = alt_rollouts
        
        top_k = rollouts[:int(HYPERPARAMETERS['N_ROLLOUTS'] * HYPERPARAMETERS['TOP_K_PERCENT'])]

        best_reward = top_k[0][1]
        
        data = get_training_data(top_k, HYPERPARAMETERS['N'], device)
        losses = [train_epoch(data, model, criterion, optimizer, HYPERPARAMETERS['N']) for _ in range(HYPERPARAMETERS['N_ITER'])]

        wandb.log({'loss': np.mean(losses), 'best_reward': best_reward})
        
        tq.set_description(f'loss.: {np.mean(losses):.4f} best reward.: {best_reward}')
        if best_reward < previous_best_reward:
            previous_best_reward = best_reward
            fig = visualize.visualize_grid(top_k[0][0][-1].view(HYPERPARAMETERS['N'], HYPERPARAMETERS['N']), f'./figures/N_{HYPERPARAMETERS["N"]}_R_{best_reward}_gen_{i}')
            wandb.log({"best_rollout": wandb.Image(plt)})
            plt.close(fig)

        if best_reward == 0:
            fig = visualize.visualize_grid(top_k[0][0][-1].view(HYPERPARAMETERS['N'], HYPERPARAMETERS['N']), f'./figures/solution/solution_{i}')
            print("Solution found!")
            wandb.log({"solution": wandb.Image(plt)})
            plt.close(fig)
            break

        rollouts = rollouts[:HYPERPARAMETERS['N_ROLLOUTS'] * 4]


    return model


def main():
    wandb.init(project=HYPERPARAMETERS["PROJECT"], entity="conjecture-team")

    runname = HYPERPARAMETERS['RUN_NAME'] if HYPERPARAMETERS['RUN_NAME'] is not None else "-".join(wandb.run.name.split("-")[:-1])
    wandb.run.name = wandb.run.name.split("-")[-1] + "-" + runname 
    wandb.run.save()

    wandb.config.update(HYPERPARAMETERS)

    trained_model = train(HYPERPARAMETERS)

    wandb.finish()


if __name__ == '__main__':
    main()
