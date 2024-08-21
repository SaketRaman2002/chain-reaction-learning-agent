import pygame
import numpy as np
import random
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import pandas as pd
import pygame
from dnq import DQNAgent
import copy
from config import params

criterion = nn.MSELoss()


# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 4
SCREEN_WIDTH, SCREEN_HEIGHT = BOARD_SIZE*100, BOARD_SIZE*100

CELL_SIZE = SCREEN_WIDTH // BOARD_SIZE
BACKGROUND_COLOR = (30, 30, 30)
PLAYER_COLORS = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to draw the game board
def draw_board(screen, board):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            cell_value = board[x, y]
            color = BACKGROUND_COLOR
            if cell_value > 0:  # Player 1 orb
                color = PLAYER_COLORS[0]
            elif cell_value < 0:  # Player 2 orb
                color = PLAYER_COLORS[1]
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            # Adjust the size of the circle based on the number of orbs
            orb_count = abs(cell_value)
            if orb_count > 0:
                # Calculate circle size (example logic, adjust as needed)
                base_size = 20  # Base size for one orb
                size_decrement = 5  # Decrease size for each additional orb
                circle_size = max(10, base_size - (orb_count - 1) * size_decrement)  # Ensure size is not too small
                
                # Draw the circle
                pygame.draw.circle(screen, (255, 255, 255), (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), circle_size)
    
    # Draw white grid lines (existing logic)
    
    # Draw white grid lines
    for x in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, (255, 255, 255), (x * CELL_SIZE, 0), (x * CELL_SIZE, SCREEN_HEIGHT))
    for y in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, (255, 255, 255), (0, y * CELL_SIZE), (SCREEN_WIDTH, y * CELL_SIZE))

# Main game class
class ChainReactionGame:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.move_count = 0

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.game_over = False
        
    def clone(self):
        return copy.deepcopy(self)   

    def is_valid_move(self, x, y):
        # Check if the cell is empty or belongs to the current player
        return self.board[x, y] == 0 or np.sign(self.board[x, y]) == self.current_player

    def get_valid_moves(self):
        valid_moves = []
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.is_valid_move(x, y):
                    valid_moves.append(x * BOARD_SIZE + y)
        return valid_moves
    
    def get_total_orbs(self):
        return np.sum(np.abs(self.board))

    def place_orb(self, x, y):
        # Determine the maximum number of orbs a cell can hold
        max_orbs = 4  # Default for non-edge cells
        if x in [0, BOARD_SIZE-1] and y in [0, BOARD_SIZE-1]:
            max_orbs = 2  # Corner cells
        elif x in [0, BOARD_SIZE-1] or y in [0, BOARD_SIZE-1]:
            max_orbs = 3  # Edge cells

        # Check if the move is valid
        if self.board[x, y] == 0 or np.sign(self.board[x, y]) == self.current_player:
            self.board[x, y] += self.current_player
            self.move_count += 1  # Increment move count
            if abs(self.board[x, y]) >= max_orbs:
                self.explode_orbs(x, y)
                if self.check_winner():
                    self.game_over = True
            self.switch_player()
        else:
            print("Invalid move")

    def explode_orbs(self, x, y):
        if self.check_winner():
            self.game_over = True
            return
        
        # Reset the current cell
        self.board[x, y] = 0

        # Define the directions of explosion
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Down, Up

        # Distribute orbs to neighboring cells and check for further explosions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                self.board[nx, ny] = (abs(self.board[nx, ny]) + 1) * (self.current_player)

                # Check if the neighboring cell needs to explode
                max_orbs_neighbor = 4
                if nx in [0, BOARD_SIZE-1] and ny in [0, BOARD_SIZE-1]:
                    max_orbs_neighbor = 2
                elif nx in [0, BOARD_SIZE-1] or ny in [0, BOARD_SIZE-1]:
                    max_orbs_neighbor = 3
                if abs(self.board[nx, ny]) >= max_orbs_neighbor:
                    self.explode_orbs(nx, ny)  

    def switch_player(self):
        self.current_player *= -1

    def check_winner(self):
        player1_orbs = 0
        player2_orbs = 0
    
        # Iterate through the board to count orbs for each player
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x, y] > 0:
                    player1_orbs += 1
                elif self.board[x, y] < 0:
                    player2_orbs += 1
    
        # Check if only one move has been played
        if self.move_count <= 1:
            return 0  # No winner yet
    
        # Determine the winner
        if player1_orbs == 0 and player2_orbs > 0:
            return -1  # Player 2 wins
        elif player2_orbs == 0 and player1_orbs > 0:
            return 1  # Player 1 wins
        else:
            return 0  # No winner yet

    def agent_move(self, agent, params, counter_games):
        if not params['train']:
            agent.epsilon = 0.01
        else:
            agent.epsilon = max(0.01, 1 - (counter_games * params['epsilon_decay_linear']))
        # Get old state
        state_old = agent.get_state(self).reshape((1, -1))  # Ensure state shape matches input size
        # Perform random actions based on agent.epsilon, or choose the action
            # Use the Q-network to predict the Q-values for each valid move
        valid_moves = self.get_valid_moves()
        best_score = float('-inf')
        best_action = None
        for move in valid_moves:
            grid_x, grid_y = divmod(move, BOARD_SIZE)
            game_copy = self.clone()  # Assuming you have a method to clone the game state
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
        self.place_orb(grid_x, grid_y)
        state_new = agent.get_state(self).reshape((1, -1))  # Ensure new state shape matches input size
        # Set reward based on the game state and moves
        reward = 0
        q_values = agent.forward(torch.tensor(state_old, dtype=torch.float32).to(DEVICE)) 
        reward = q_values.max().item()
        # Train short memory
        agent.train_short_memory(state_old, action, reward, state_new, self.game_over)
        # Remember the experience
        agent.remember(state_old, action, reward, state_new, self.game_over)        

    def run(self):
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        running = True
    
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    self.place_orb(grid_x, grid_y)
                    
                    # Check for a winner after placing an orb
                    winner = self.check_winner()
                    if winner != 0:
                        print(f"Player {winner} wins!")
                        running = False
    
            screen.fill(BACKGROUND_COLOR)
            draw_board(screen, self.board)
            pygame.display.flip()
            clock.tick(60)
    
        pygame.quit()

    def AgentRun(self, agent, params):
        pygame.init()
        agent = DQNAgent(params)
        agent = agent.to(DEVICE)
        agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        running = True
        counter_games = 0
        player1_states = []
        player2_states = []
        state_scores = {}
        move_count = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.current_player == 1:
                    x, y = pygame.mouse.get_pos()
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    if self.is_valid_move(grid_x, grid_y):
                        self.place_orb(grid_x, grid_y)
            if self.current_player == -1 and not self.game_over:
                self.agent_move(agent, params, counter_games)
            screen.fill(BACKGROUND_COLOR)
            draw_board(screen, self.board)
            pygame.display.flip()
            clock.tick(60)
            winner = self.check_winner()
            if winner != 0:
                print(f"Player {winner} wins!")
                running = False
            if params['train']:
                agent.replay_new(agent.memory, params['batch_size'])
            winner = self.check_winner()
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
        pygame.quit()            



# Run the game
if __name__ == "__main__":
    if params['train']:
        agent = DQNAgent(params)
        if params["load_weights"]:
            agent.load_state_dict(torch.load(params["weights_path"], map_location=DEVICE))
        agent.train()  # Set the agent to training mode

        game = ChainReactionGame()
        game.AgentRun(agent, params)

        # Save the updated weights after the game
        torch.save(agent.state_dict(), params["weights_path"])
    else:
        game = ChainReactionGame()
        game.run()