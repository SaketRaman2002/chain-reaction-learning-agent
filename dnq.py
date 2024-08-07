import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import pandas as pd

BOARD_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()

        

    def network(self):
        self.fc1 = nn.Linear(BOARD_SIZE*BOARD_SIZE, self.first_layer)
        self.fc2 = nn.Linear(self.first_layer, self.second_layer)
        self.fc3 = nn.Linear(self.second_layer, self.third_layer)
        self.fc4 = nn.Linear(self.third_layer, BOARD_SIZE*BOARD_SIZE)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


    def get_state(self, game):
        board_state = game.board
        state = []
        for row in board_state:
            for cell in row:
                state.append(cell)  # number of orbs and player ID
        return np.array(state)

# def set_reward(self, game, moves):
#     """
#     Update the score of each state based on the number of moves it took to win the game.
#     """
#     winner = game.check_winner()
#     if winner == self.agent_target:
#         score = 1 / moves  # Higher score for fewer moves
#     else:
#         score = -1 / moves  # Lower score for more moves

#     for state in self.actual:
#         state_tuple = tuple(state)
#         self.state_scores[state_tuple] += score


    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory ,batch_size):
        """
        Train the agent on experiences from replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        # print(minibatch)    
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][0, action] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.forward(torch.tensor(next_state, dtype=torch.float32)).detach())
        output = self.forward(torch.tensor(state, dtype=torch.float32))
        target_f = output.clone()
        target_f[0, action] = target
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

# Initialize your DQNAgent with appropriate parameters and integrate it with your Chain Reaction game loop.
