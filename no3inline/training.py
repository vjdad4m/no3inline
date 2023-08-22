import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import no3inline
import wandb
import visualize
import matplotlib.pyplot as plt
from config import HYPERPARAMETERS
class Generator(nn.Module):
    def __init__(self, N):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1 , 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(N * N, N * N)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return torch.softmax(x, dim=-1)

def calculate_rewards_per_state(state_list, N, reward_type):
    if reward_type == 'summed':
        rewards = []
        for state in state_list:
            reward = no3inline.calculate_reward(state.view(N, N))
        rewards.append(reward)
        reward = np.sum(rewards)
    elif reward_type == 'laststate':
        state = state_list[-1]
        reward = no3inline.calculate_reward(state.view(N, N))
    return reward

def generate_rollout(model, N, device, reward_type):
    states = [torch.zeros((N * N)).float()]
    
    for _ in range(2 * N):
        next_state_probabilities = model(states[-1].view((1, N, N)).to(device)).cpu().detach()[0]
        next_state_probabilities[states[-1] == 1.0] = float('-inf')
        next_state_probabilities = torch.softmax(torch.flatten(next_state_probabilities), dim=-1)
        
        action = torch.multinomial(next_state_probabilities, num_samples=1).item()
        new_state = states[-1].clone()
        new_state[action] = 1
        
        states.append(new_state)
    rewards = calculate_rewards_per_state(states, N, reward_type)
    return states, np.sum(rewards)

def generate_batched_rollout(model, N, BATCH_SIZE, device, reward_type):
    states = [torch.zeros((BATCH_SIZE, N * N)).float()]

    for _ in range(2 * N):
        next_state_probabilities = model(states[-1].reshape((BATCH_SIZE, 1, N, N)).to(device)).cpu().detach()
        mask = states[-1] == 1.0

        next_state_probabilities[mask] = float('-inf')
        next_state_probabilities = torch.softmax(next_state_probabilities, dim=1)
        action = torch.multinomial(next_state_probabilities, num_samples=1).squeeze()
        new_state = states[-1].clone()
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
        output = model(X.view((2 * N, 1, N, N))).view((2 * N, N, N))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)

def train(HYPERPARAMETERS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device = }')

    model = Generator(HYPERPARAMETERS['N']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()

    rollouts = []
    tq = tqdm.trange(HYPERPARAMETERS['N_EPOCHS'])
    for i in tq:
        # - Simple rollout generation
        # rollouts.extend([generate_rollout(model, HYPERPARAMETERS['N'], device, HYPERPARAMETERS['REWARD_TYPE']) 
        #                 for _ in range(HYPERPARAMETERS['N_ROLLOUTS'])])

        # - Batched rollout generation
        rollouts.extend(generate_batched_rollout(model, HYPERPARAMETERS['N'], HYPERPARAMETERS['N_ROLLOUTS'], device, HYPERPARAMETERS['REWARD_TYPE']))
    
        rollouts.sort(key=lambda x: x[1])
        
        top_k = rollouts[:int(HYPERPARAMETERS['N_ROLLOUTS'] * HYPERPARAMETERS['TOP_K_PERCENT'])]
        
        # Uncomment if using simple rollout generation
        # top_k = torch.stack(top_k)

        best_reward = top_k[0][1]
        
        data = get_training_data(top_k, HYPERPARAMETERS['N'], device)
        losses = [train_epoch(data, model, criterion, optimizer, HYPERPARAMETERS['N']) for _ in range(HYPERPARAMETERS['N_ITER'])]

        wandb.log({'loss': np.mean(losses), 'best_reward': best_reward})
        
        tq.set_description(f'loss.: {np.mean(losses):.4f} best reward.: {best_reward}')
        if i % 10 == 0:
            fig=visualize.visualize_grid(top_k[0][0][-1].view(HYPERPARAMETERS['N'], HYPERPARAMETERS['N']), f'./figures/gen_{i}')
            wandb.log({f"gen{i}":wandb.Image(plt)})
            


        rollouts = rollouts[:HYPERPARAMETERS['N'] * HYPERPARAMETERS['N_ROLLOUTS'] * 4]


    return model


def main():
    wandb.init(project="testing", entity="conjecture-team")

    runname = HYPERPARAMETERS['RUN_NAME'] if HYPERPARAMETERS['RUN_NAME'] is not None else "-".join(wandb.run.name.split("-")[:-1])
    wandb.run.name = wandb.run.name.split("-")[-1] + "-" + runname 
    wandb.run.save()

    wandb.config.update(HYPERPARAMETERS)

    trained_model = train(HYPERPARAMETERS)

    wandb.finish()


if __name__ == '__main__':
    main()