from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import visualize
from config import HYPERPARAMETERS

import no3inline
import wandb


# Define the neural network (Generator) that acts as the policy in reinforcement learning
class Generator(nn.Module):
    def __init__(self, N: int):
        super(Generator, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1 , 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # Flatten and linear layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(N * N, N * N)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    # Forward pass of the neural network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applying convolutional layers with ReLU activation functions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return torch.softmax(x, dim=-1)  # Softmax for probability distribution

def calculate_rewards_per_state(state_list: List[torch.Tensor], N: int, reward_type: str) -> float:
    # Check the type of reward calculation requested
    if reward_type == 'summed':
        # Initialize an empty list to store rewards
        rewards = []
        # Iterate over each state in the state list
        for state in state_list:
            # Calculate the reward for the current state using the no3inline function
            # and reshape the state into a 2D (N x N) tensor
            reward += no3inline.calculate_reward(state.view(N, N))
        # Append the calculated reward of the last state to the rewards list
        rewards.append(reward)
        # Sum up all the rewards
        reward = np.sum(rewards)
    # If the reward type is 'laststate'
    elif reward_type == 'laststate':
        # Take the last state from the state list
        state = state_list[-1]
        # Calculate the reward for the last state
        reward = no3inline.calculate_reward(state.view(N, N))
    # Return the final calculated reward
    return reward

def generate_rollout(model, N: int, device: str, reward_type: str) -> Tuple[List[torch.Tensor], float]:
    # Initialize a list of states starting with an all-zero tensor of size (N * N)
    states = [torch.zeros((N * N)).float()]

    # Generate a sequence of states; the loop runs for 2*N iterations
    for _ in range(2 * N):
        # Pass the last state through the model to get next state probabilities
        # Reshape it to (1, N, N) before passing to the model
        # Move to the specified device (usually GPU or CPU)
        # Then, get it back to CPU and detach it from the computation graph
        next_state_probabilities = model(states[-1].view((1, N, N)).to(device)).cpu().detach()[0]
        
        # Mask positions in the next_state_probabilities tensor where the last state is 1.0 
        # (i.e., where a decision has already been made) to -infinity
        next_state_probabilities[states[-1] == 1.0] = float('-inf')
        
        # Flatten the next_state_probabilities tensor and apply the softmax function to get a probability distribution
        next_state_probabilities = torch.softmax(torch.flatten(next_state_probabilities), dim=-1)
        
        # Sample an action based on the probability distribution obtained above
        action = torch.multinomial(next_state_probabilities, num_samples=1).item()
        
        # Clone the last state to create a new one
        new_state = states[-1].clone()
        # Update the chosen position (action) in the new state to 1.0
        new_state[action] = 1
        
        # Append the new state to the list of states
        states.append(new_state)

    # Calculate rewards for the entire sequence of states based on the specified reward_type
    rewards = calculate_rewards_per_state(states, N, reward_type)
    
    # Return the list of states and the total rewards
    return states, np.sum(rewards)

def generate_batched_rollout(model, N: int, BATCH_SIZE: int, device: str, reward_type: str) -> List[Tuple[torch.Tensor, float]]:
    # Initialize a list of states for each batch, starting with a tensor of zeros of size (BATCH_SIZE, N * N)
    states = [torch.zeros((BATCH_SIZE, N * N)).float()]

    # Generate a sequence of states for each batch; the loop runs for 2*N iterations
    for _ in range(2 * N):
        # Reshape the last state to (BATCH_SIZE, 1, N, N) and pass it through the model to get next state probabilities
        # Move to the specified device (usually GPU or CPU)
        # Then, get it back to CPU and detach it from the computation graph
        next_state_probabilities = model(states[-1].reshape((BATCH_SIZE, 1, N, N)).to(device)).cpu().detach()

        # Create a mask where the last state is 1.0 (i.e., a decision has already been made)
        mask = states[-1] == 1.0

        # Set the probabilities in next_state_probabilities to -infinity where the mask is True
        next_state_probabilities[mask] = float('-inf')

        # Apply the softmax function across the channel dimension to get a probability distribution for each batch
        next_state_probabilities = torch.softmax(next_state_probabilities, dim=1)

        # Sample an action for each batch based on the obtained probability distributions
        action = torch.multinomial(next_state_probabilities, num_samples=1).squeeze()

        # Clone the last state to create a new state for each batch
        new_state = states[-1].clone()
        
        # Update the chosen positions (actions) in the new states to 1.0 for each batch
        new_state[torch.arange(BATCH_SIZE), action] = 1

        # Append the new states to the list of states
        states.append(new_state)

    # Stack all states together and reshape them for further processing
    rollouts = torch.stack(states).reshape((2 * N + 1, BATCH_SIZE, N * N))
    
    # Change the order of dimensions for rollouts from (Sequence Length, Batch Size, State Size) to (Batch Size, Sequence Length, State Size)
    rollouts = rollouts.permute((1, 0, 2))

    # Calculate rewards for each rollout based on the specified reward_type
    rewards = [calculate_rewards_per_state(rollout, N, reward_type) for rollout in rollouts]
    
    # Return the rollouts paired with their associated rewards
    return list(zip(rollouts, rewards))

def tensor_from_rollout(rollout: torch.Tensor, N: int) -> torch.Tensor:
    # Reshape the rollout tensor to have dimensions (length of rollout, N, N)
    stack = rollout.view(len(rollout), N, N)

    # Initialize a tensor of zeros with shape (2, len(rollout) - 1, N, N)
    # This will hold the 'current' states and their 'next' states side by side
    tensor_list = torch.zeros((2, len(rollout) - 1, N, N))

    # Set the first dimension of tensor_list (at index 0) to all states of the rollout except the last one
    tensor_list[0] = stack[:-1]

    # Set the second dimension of tensor_list (at index 1) to all states of the rollout starting from the second one
    tensor_list[1] = stack[1:]

    # Return the tensor containing pairs of 'current' and 'next' states
    return tensor_list

def get_training_data(top_k: List[Tuple[torch.Tensor, float]], N: int, device: str) -> List[torch.Tensor]:
    # For each rollout in the top performing rollouts (top_k),
    # - Convert the rollout to the tensor format capturing pairs of consecutive states using tensor_from_rollout function
    # - Move the resulting tensor to the specified device (CPU or CUDA)
    return [
        tensor_from_rollout(rollout[0], N).to(device)
        for rollout in top_k
    ]

def train_epoch(data: List[Tuple[torch.Tensor, torch.Tensor]], model, criterion: nn.Module, optimizer: torch.optim.Optimizer, N: int) -> float:
    # Initialize a list to store the loss values for each batch in the epoch
    epoch_loss = []

    # Iterate through each batch of data in the dataset
    for X, y in data:
        # Zero out any gradients from the previous iteration to ensure they don't accumulate
        optimizer.zero_grad()

        # Pass the input data (X) through the model to get the model's predictions
        # First, reshape the input data to the expected shape for the model
        # After getting the output, reshape it back to the original shape
        output = model(X.view((2 * N, 1, N, N))).view((2 * N, N, N))

        # Calculate the loss between the model's predictions (output) and the true values (y) using the provided criterion
        loss = criterion(output, y)

        # Backpropagate the error through the network to calculate gradients for each parameter
        loss.backward()

        # Update the model's parameters based on the gradients calculated in the previous step
        optimizer.step()

        # Append the loss for this batch to the epoch_loss list
        epoch_loss.append(loss.item())

    # Return the average loss for the entire epoch
    return np.mean(epoch_loss)

def train(HYPERPARAMETERS: dict):
    # Decide the computation device based on GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device = }')

    # Initialize the model and move it to the computation device
    model = Generator(HYPERPARAMETERS['N']).to(device)

    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()

    # Initialize an empty list to store the rollouts
    rollouts = []

    # Initialize a progress bar for number of epochs
    tq = tqdm.trange(HYPERPARAMETERS['N_EPOCHS'])
    for i in tq:
        # Commented-out simple rollout generation method
        # rollouts.extend([generate_rollout(model, HYPERPARAMETERS['N'], device, HYPERPARAMETERS['REWARD_TYPE']) 
        #                 for _ in range(HYPERPARAMETERS['N_ROLLOUTS'])])

        # Generate batched rollouts and add them to the rollouts list
        rollouts.extend(generate_batched_rollout(model, HYPERPARAMETERS['N'], HYPERPARAMETERS['N_ROLLOUTS'], device, HYPERPARAMETERS['REWARD_TYPE']))
    
        # Sort rollouts based on the reward
        rollouts.sort(key=lambda x: x[1])

        # Select the top 'k' rollouts based on the reward
        top_k = rollouts[:int(HYPERPARAMETERS['N_ROLLOUTS'] * HYPERPARAMETERS['TOP_K_PERCENT'])]

        # Convert top_k into a tensor if using simple rollout generation (currently commented out)
        # top_k = torch.stack(top_k)

        # Get the reward of the best rollout
        best_reward = top_k[0][1]

        # Convert top_k rollouts into training data
        data = get_training_data(top_k, HYPERPARAMETERS['N'], device)

        # Train the model for 'N_ITER' times and store the loss for each iteration
        losses = [train_epoch(data, model, criterion, optimizer, HYPERPARAMETERS['N']) for _ in range(HYPERPARAMETERS['N_ITER'])]

        # Log loss and best reward values using wandb (Weights & Biases tool for experiment tracking)
        wandb.log({'loss': np.mean(losses), 'best_reward': best_reward})

        # Update the progress bar description with loss and best reward
        tq.set_description(f'loss.: {np.mean(losses):.4f} best reward.: {best_reward}')

        # Log an image of the best rollout after every 10 epochs
        if i % 10 == 0:
            fig = visualize.visualize_grid(top_k[0][0][-1].view(HYPERPARAMETERS['N'], HYPERPARAMETERS['N']), f'./figures/gen_{i}')
            wandb.log({"best_rollout": wandb.Image(plt)})

        # Trim the rollouts list to retain only the recent ones
        rollouts = rollouts[:HYPERPARAMETERS['N_ROLLOUTS'] * 4]

    # Return the trained model
    return model

def main():
    # Initialize a wandb (Weights & Biases) run with a specified project and entity
    wandb.init(project="testing", entity="conjecture-team")

    # Check if a run name is given in the hyperparameters, 
    # if not, construct one from the wandb default run name
    runname = HYPERPARAMETERS['RUN_NAME'] if HYPERPARAMETERS['RUN_NAME'] is not None else "-".join(wandb.run.name.split("-")[:-1])
    
    # Reformat the wandb run name by appending the constructed run name
    wandb.run.name = wandb.run.name.split("-")[-1] + "-" + runname 
    
    # Save the current state of the wandb run 
    wandb.run.save()

    # Update the wandb configuration with the hyperparameters
    wandb.config.update(HYPERPARAMETERS)

    # Train the model using the hyperparameters
    trained_model = train(HYPERPARAMETERS)

    # Finish the current wandb run, finalizing all logs and data
    wandb.finish()


# This checks if the script is being run directly (not imported), 
# and if so, calls the main() function
if __name__ == '__main__':
    main()
