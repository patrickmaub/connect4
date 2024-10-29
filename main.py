import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pygame
import math
import random
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
import os
import shutil

# Constants
BOARD_ROWS = 6
BOARD_COLS = 7
WINDOW_SIZE = 4
EMPTY = 0
WIN_SCORE = 1.0
DRAW_SCORE = 0.5
LOSS_SCORE = 0.0

class Connect4State:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        
    def copy(self):
        new_state = Connect4State()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.last_move = self.last_move
        return new_state
    
    def get_valid_moves(self):
        return [col for col in range(BOARD_COLS) if self.board[0][col] == 0]
    
    def make_move(self, col):
        for row in range(BOARD_ROWS-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                return True
        return False
    
    def check_win(self):
        # Check horizontal
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS-3):
                window = self.board[row, col:col+4]
                if np.all(window == 1):
                    return 1
                elif np.all(window == 2):
                    return 2
                    
        # Check vertical
        for row in range(BOARD_ROWS-3):
            for col in range(BOARD_COLS):
                window = self.board[row:row+4, col]
                if np.all(window == 1):
                    return 1
                elif np.all(window == 2):
                    return 2
                    
        # Check diagonal (positive slope)
        for row in range(BOARD_ROWS-3):
            for col in range(BOARD_COLS-3):
                window = [self.board[row+i][col+i] for i in range(4)]
                if all(x == 1 for x in window):
                    return 1
                elif all(x == 2 for x in window):
                    return 2
                    
        # Check diagonal (negative slope)
        for row in range(3, BOARD_ROWS):
            for col in range(BOARD_COLS-3):
                window = [self.board[row-i][col+i] for i in range(4)]
                if all(x == 1 for x in window):
                    return 1
                elif all(x == 2 for x in window):
                    return 2
                    
        # Check for draw
        if len(self.get_valid_moves()) == 0:
            return 0
            
        return None

    def get_state_tensor(self):
        tensor = torch.zeros((3, BOARD_ROWS, BOARD_COLS))
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i][j] == 1:
                    tensor[0][i][j] = 1
                elif self.board[i][j] == 2:
                    tensor[1][i][j] = 1
        tensor[2].fill_(1 if self.current_player == 1 else 0)
        return tensor

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        
        # Common layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 32, 1)
        self.policy_fc = nn.Linear(32 * BOARD_ROWS * BOARD_COLS, BOARD_COLS)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 32, 1)
        self.value_fc1 = nn.Linear(32 * BOARD_ROWS * BOARD_COLS, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Common layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * BOARD_ROWS * BOARD_COLS)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * BOARD_ROWS * BOARD_COLS)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class MCTS:
    def __init__(self, net, num_simulations=800, c_puct=1.0):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.Qsa = defaultdict(float)  # Q values for state-action pairs
        self.Nsa = defaultdict(int)    # Number of visits for state-action pairs
        self.Ns = defaultdict(int)     # Number of visits for states
        self.Ps = {}                   # Initial policy probabilities for states
        self.valid_moves = {}          # Valid moves for states
        self.states = {}               # Game states
        
    def get_action_prob(self, state, temp=1):
        for _ in range(self.num_simulations):
            self.search(state)
            
        s = self._get_state_key(state)
        counts = [self.Nsa[(s, a)] for a in range(BOARD_COLS)]
        
        if temp == 0:
            best_moves = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_move = np.random.choice(best_moves)
            probs = np.zeros(BOARD_COLS)
            probs[best_move] = 1
            return probs
            
        counts = [x ** (1. / temp) for x in counts]
        total = sum(counts)
        probs = [x / total if total > 0 else 0 for x in counts]
        return probs
        
    def search(self, state):
        s = self._get_state_key(state)
        
        if s not in self.states:
            self.states[s] = state.copy()
            
        if state.check_win() is not None:
            return -self._get_game_ended_value(state)
            
        if s not in self.Ps:
            self.valid_moves[s] = state.get_valid_moves()
            
            # Neural network evaluation
            state_tensor = state.get_state_tensor().unsqueeze(0)
            with torch.no_grad():
                log_ps, v = self.net(state_tensor)
                self.Ps[s] = torch.exp(log_ps).cpu().numpy()[0]
                
            # Mask invalid moves
            valid_moves_mask = np.zeros(BOARD_COLS)
            valid_moves_mask[self.valid_moves[s]] = 1
            self.Ps[s] = self.Ps[s] * valid_moves_mask
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                self.Ps[s] = valid_moves_mask / np.sum(valid_moves_mask)
                
            return -v.item()
            
        cur_best = -float('inf')
        best_act = -1
        
        for a in self.valid_moves[s]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s])
                
            if u > cur_best:
                cur_best = u
                best_act = a
                
        a = best_act 
        next_state = state.copy()
        next_state.make_move(a)
        
        v = self.search(next_state)
        
        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1
        return -v
        
    def _get_state_key(self, state):
        return state.board.tobytes() + bytes([state.current_player])
        
    def _get_game_ended_value(self, state):
        winner = state.check_win()
        if winner is None:
            return 0
        if winner == 0:
            return DRAW_SCORE
        return WIN_SCORE if winner == state.current_player else LOSS_SCORE

class Connect4GUI:
    def __init__(self, cell_size=100):
        pygame.init()
        self.cell_size = cell_size
        self.width = BOARD_COLS * cell_size
        self.height = (BOARD_ROWS + 1) * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Connect 4 AI')
        self.colors = {
            0: (255, 255, 255),  # Empty
            1: (255, 0, 0),      # Player 1 (Red)
            2: (255, 255, 0)     # Player 2 (Yellow)
        }
        
    def draw_board(self, state):
        self.screen.fill((0, 0, 255))  # Blue background
        
        # Draw pieces
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                pygame.draw.circle(
                    self.screen,
                    self.colors[state.board[row][col]],
                    (col * self.cell_size + self.cell_size//2,
                     (row + 1) * self.cell_size + self.cell_size//2),
                    self.cell_size//2 - 5
                )
                
        pygame.display.update()
        
    def get_column_from_mouse(self, pos):
        x = pos[0]
        return x // self.cell_size

def self_play(net, num_games=100, mcts_simulations=800):
    mcts = MCTS(net, num_simulations=mcts_simulations)
    examples = []
    
    for i in range(num_games):
        state = Connect4State()
        current_examples = []
        
        while True:
            temp = 1.0 if len(state.get_valid_moves()) > 10 else 0.1
            pi = mcts.get_action_prob(state, temp)
            
            current_examples.append([
                state.get_state_tensor(),
                torch.FloatTensor(pi),
                state.current_player
            ])
            
            action = np.random.choice(len(pi), p=pi)
            state.make_move(action)
            
            game_result = state.check_win()
            if game_result is not None:
                # Process results
                r = 0 if game_result == 0 else (1 if game_result == 1 else -1)
                for example in current_examples:
                    player = example[2]
                    value = r if player == 1 else -r
                    examples.append([
                        example[0],
                        example[1],
                        torch.FloatTensor([value])
                    ])
                break
                
        print(f'Self-play game {i+1}/{num_games} completed')
        
    return examples
# [Previous code remains the same until train_network function]

def train_network(net, examples, num_epochs=10, batch_size=32, lr=0.001, writer=None):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    pi_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()
    
    local_writer = writer
    if local_writer is None:
        local_writer = SummaryWriter('runs/connect4_training', flush_secs=1)
        
    global_step = 0
    
    for epoch in range(num_epochs):
        net.train()
        total_pi_loss = 0
        total_v_loss = 0
        batch_count = 0
        
        indices = np.arange(len(examples))
        np.random.shuffle(indices)
        
        for i in range(0, len(examples), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [examples[idx] for idx in batch_indices]
            
            state_batch = torch.stack([x[0] for x in batch])
            pi_batch = torch.stack([x[1] for x in batch])
            v_batch = torch.stack([x[2] for x in batch])
            
            optimizer.zero_grad()
            
            out_pi, out_v = net(state_batch)
            pi_loss = pi_criterion(out_pi, pi_batch)
            v_loss = v_criterion(out_v.squeeze(), v_batch.squeeze())
            total_loss = pi_loss + v_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Log batch-level metrics
            local_writer.add_scalar('Loss/Policy/Step', pi_loss.item(), global_step)
            local_writer.add_scalar('Loss/Value/Step', v_loss.item(), global_step)
            local_writer.add_scalar('Loss/Total/Step', total_loss.item(), global_step)
            
            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()
            batch_count += 1
            global_step += 1
        
        # Log epoch-level metrics
        avg_pi_loss = total_pi_loss / batch_count
        avg_v_loss = total_v_loss / batch_count
        
        local_writer.add_scalar('Loss/Policy/Epoch', avg_pi_loss, epoch)
        local_writer.add_scalar('Loss/Value/Epoch', avg_v_loss, epoch)
        local_writer.add_scalar('Loss/Total/Epoch', avg_pi_loss + avg_v_loss, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average policy loss: {avg_pi_loss:.4f}')
        print(f'Average value loss: {avg_v_loss:.4f}')
    
    local_writer.flush()

def train_model(net, num_iterations=100, num_episodes=100, mcts_simulations=800):
    """Training pipeline for the Connect 4 AI"""
    # Clean up old event files
    if os.path.exists('runs/connect4_training'):
        shutil.rmtree('runs/connect4_training')
    
    writer = SummaryWriter('runs/connect4_training', flush_secs=1)
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Self-play phase
        examples = self_play(net, num_games=num_episodes, mcts_simulations=mcts_simulations)
        
        # Training phase
        train_network(net, examples, writer=writer)
        
        # Save model checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'iteration': iteration,
            'model_state_dict': net.state_dict(),
        }, f'checkpoints/connect4_model_iter_{iteration}.pt')
        
        # Save latest model
        torch.save({
            'iteration': iteration,
            'model_state_dict': net.state_dict(),
        }, 'checkpoints/connect4_model_latest.pt')
        
        # Evaluation phase (optional)
        if iteration > 0 and iteration % 5 == 0:
            prev_net = Connect4Net()
            prev_net.load_state_dict(torch.load(f'checkpoints/connect4_model_iter_{iteration-5}.pt')['model_state_dict'])
            
            wins = 0
            num_eval_games = 40
            
            for i in range(num_eval_games):
                player1 = AIPlayer(net, mcts_simulations=400)
                player2 = AIPlayer(prev_net, mcts_simulations=400)
                
                if i % 2 == 0:  # Alternate colors
                    result = play_game(player1, player2)
                    wins += 1 if result == 1 else (0.5 if result == 0 else 0)
                else:
                    result = play_game(player2, player1)
                    wins += 1 if result == 2 else (0.5 if result == 0 else 0)
            
            win_rate = wins / num_eval_games
            writer.add_scalar('Evaluation/WinRate', win_rate, iteration)
            print(f"Evaluation Win Rate: {win_rate:.3f}")
    
    writer.close()

class AIPlayer:
    def __init__(self, net, mcts_simulations=800, temperature=0.1):
        self.net = net
        self.mcts = MCTS(net, num_simulations=mcts_simulations)
        self.temperature = temperature
    
    def get_move(self, state):
        pi = self.mcts.get_action_prob(state, temp=self.temperature)
        return np.argmax(pi)

class HumanPlayer:
    def __init__(self, gui):
        self.gui = gui
    
    def get_move(self, state):
        valid_moves = state.get_valid_moves()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    col = self.gui.get_column_from_mouse(event.pos)
                    if col in valid_moves:
                        return col
            pygame.time.wait(100)

def play_game(player1, player2, gui=None, delay=500):
    state = Connect4State()
    players = {1: player1, 2: player2}
    
    if gui:
        gui.draw_board(state)
    
    while True:
        current_player = players[state.current_player]
        move = current_player.get_move(state)
        
        if move is None:  # Game was quit
            return None
        
        state.make_move(move)
        
        if gui:
            gui.draw_board(state)
            pygame.time.wait(delay)
        
        result = state.check_win()
        if result is not None:
            return result

def test_tensorboard():
    """Test function to verify TensorBoard logging"""
    if os.path.exists('runs/connect4_training'):
        shutil.rmtree('runs/connect4_training')
    
    writer = SummaryWriter('runs/connect4_training')
    for i in range(10):
        writer.add_scalar('Test/Value', i * i, i)
    writer.flush()
    writer.close()
    print("Test data written to TensorBoard. Please check http://localhost:6006")

def main():
    # Initialize GUI
    gui = Connect4GUI()
    
    # Initialize network
    net = Connect4Net()
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check for existing checkpoints
    try:
        checkpoint = torch.load('checkpoints/connect4_model_latest.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded existing model")
    except FileNotFoundError:
        print("Starting with fresh model")
    
    while True:
        print("\nConnect 4 Menu:")
        print("1. Train Model")
        print("2. Play Against AI")
        print("3. Watch AI vs AI")
        print("4. Test TensorBoard")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            num_iterations = int(input("Enter number of iterations (default 100): ") or "100")
            num_episodes = int(input("Enter number of episodes per iteration (default 100): ") or "100")
            train_model(net, num_iterations=num_iterations, num_episodes=num_episodes)
        
        elif choice == '2':
            human = HumanPlayer(gui)
            ai = AIPlayer(net)
            
            # Randomly decide who goes first
            if random.random() < 0.5:
                result = play_game(human, ai, gui)
                player_color = "Red"
            else:
                result = play_game(ai, human, gui)
                player_color = "Yellow"
            
            print(f"You played as {player_color}")
            
            if result == 0:
                print("Game ended in a draw!")
           
            else:
                print("AI won!")
        
        elif choice == '3':
            ai1 = AIPlayer(net, temperature=0.1)
            ai2 = AIPlayer(net, temperature=0.1)
            result = play_game(ai1, ai2, gui, delay=500)
            
            if result == 0:
                print("Game ended in a draw!")
            else:
                print(f"AI {result} won!")
        
        elif choice == '4':
            test_tensorboard()
        
        elif choice == '5':
            break
        
        else:
            print("Invalid choice. Please try again.")
    
    pygame.quit()

if __name__ == "__main__":
    main()