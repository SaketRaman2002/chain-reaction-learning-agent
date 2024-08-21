import pygame
import torch
import numpy as np
import random
from random import randint
from torch import optim
from game import ChainReactionGame
from dnq import DQNAgent  # Assuming your DQNAgent class is defined in dnq.py

import torch
import torch.optim as optim
import torch.nn as nn

from config import params

# Assuming agent is your model
criterion = nn.MSELoss()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 4



def run(params):
    """
    Run the DQN algorithm based on the parameters previously set.
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    
    counter_games = 0
    state_scores = {}  # Dictionary to maintain scores of states

    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialize game
        game = ChainReactionGame()
        state = agent.get_state(game).reshape((1, -1))  # Ensure state shape matches input size

        done = False
        move_count = 0
        player1_states = []
        player2_states = [] # List to track states encountered by the user
        while not done:
            if not params['train']:
                agent.epsilon = 0.01
            else:
                agent.epsilon = max(0.01, 1 - (counter_games * params['epsilon_decay_linear']))

            # Get old state
            state_old = state

            # Perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                if len(game.get_valid_moves()) == 0:
                    print(f'No valid moves left. Ending game. {state_old}')
                else:    
                    action = random.choice(game.get_valid_moves())
            else:
                # Use the Q-network to predict the Q-values for each valid move
                valid_moves = game.get_valid_moves()
                best_score = float(0)
                best_action = None
                for move in valid_moves:
                    grid_x, grid_y = divmod(move, BOARD_SIZE)
                    game_copy = game.clone()  # Assuming you have a method to clone the game state
                    game_copy.place_orb(grid_x, grid_y)
                    next_state = agent.get_state(game_copy).reshape((1, -1))
                    q_values = agent.forward(torch.tensor(next_state, dtype=torch.float32).to(DEVICE))
                    score = q_values.max().item()
                    if score > best_score:
                        best_score = score
                        best_action = move
            
                action = best_action if best_action is not None else random.choice(valid_moves)

            # Perform new move
            grid_x, grid_y = divmod(action, BOARD_SIZE)
            game.place_orb(grid_x, grid_y)
            state_new = agent.get_state(game).reshape((1, -1))  # Ensure new state shape matches input size

            # Set reward based on the game state and moves
            reward = 0
            q_values = agent.forward(torch.tensor(state_old, dtype=torch.float32).to(DEVICE)) 
            reward= q_values.max().item()

            # Train short memory
            agent.train_short_memory(state_old, action, reward, state_new, done)

            # Remember the experience
            agent.remember(state_old, action, reward, state_new, done)

            state = state_new

            if game.current_player == 1:
                player1_states.append((state_old, move_count))
            else:
                player2_states.append((state_old, move_count))

            done = game.check_winner() != 0



            if params['display']:
                # Add display logic if needed
                pass

            move_count += 1

        # Replay new experiences
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])

        winner = game.check_winner()
        if winner != 0:
            total_moves = move_count
            if winner == 1:
                # Player 1 is the winner
                for state, moves in player1_states:
                    state_key = tuple(state.flatten())
                    state_scores[state_key] = 1 / (total_moves - moves)
                for state, moves in player2_states:
                    state_key = tuple(state.flatten())
                    state_scores[state_key] = -1 / (total_moves - moves)
            else:
                # Player 2 is the winner
                for state, moves in player2_states:
                    state_key = tuple(state.flatten())
                    state_scores[state_key] = 1 / (total_moves - moves)
                for state, moves in player1_states:
                    state_key = tuple(state.flatten())
                    state_scores[state_key] = -1 / (total_moves - moves)

            # Training the model using state_scores
            agent.train()
            for state_key, score in state_scores.items():
                state_tensor = torch.tensor(state_key, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                target = torch.tensor([score], dtype=torch.float32)

                agent.optimizer.zero_grad()
                output = agent(state_tensor)
                loss = criterion(output, target)
                loss.backward()
                agent.optimizer.step()


        counter_games += 1

    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])


    return state_scores

if __name__ == "__main__":
    run(params)