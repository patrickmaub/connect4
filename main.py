import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pygame
import math
import random
from collections import defaultdict
import time
import os
import shutil
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

# Constants
BOARD_ROWS = 6
BOARD_COLS = 7
WINDOW_SIZE = 4
EMPTY = 0
WIN_SCORE = 1.0
DRAW_SCORE = 0.5
LOSS_SCORE = 0.0

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
    torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cudnn
    cudnn.benchmark = True                        # Enable cudnn autotuner
    torch.cuda.set_device(0)                      # Ensure we're using GPU 0

    # Print GPU info
    gpu_properties = torch.cuda.get_device_properties(0)
    print(f"Using GPU: {gpu_properties.name}")
    print(f"GPU Memory: {gpu_properties.total_memory / 1024**3:.2f} GB")
    print(f"CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

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
        for row in range(BOARD_ROWS - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                return True
        return False

    def check_win(self):
        # Check horizontal
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                window = self.board[row, col:col + 4]
                if np.all(window == 1):
                    return 1
                elif np.all(window == 2):
                    return 2

        # Check vertical
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS):
                window = self.board[row:row + 4, col]
                if np.all(window == 1):
                    return 1
                elif np.all(window == 2):
                    return 2

        # Check diagonal (positive slope)
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS - 3):
                window = [self.board[row + i][col + i] for i in range(4)]
                if all(x == 1 for x in window):
                    return 1
                elif all(x == 2 for x in window):
                    return 2

        # Check diagonal (negative slope)
        for row in range(3, BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                window = [self.board[row - i][col + i] for i in range(4)]
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
        # Move input tensor to same device as model
        x = x.to(next(self.parameters()).device)

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
    def __init__(self, net, num_simulations=1600, c_puct=1.0):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.Qsa = defaultdict(float)
        self.Nsa = defaultdict(int)
        self.Ns = defaultdict(int)
        self.Ps = {}
        self.valid_moves = {}
        self.states = {}
        self.Es = {}
        self.Vs = {}

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

        if s not in self.Es:
            self.Es[s] = state.check_win()

        if self.Es[s] is not None:
            if self.Es[s] == 0:
                return 0
            else:
                return -1 if self.Es[s] == state.current_player else 1

        if s not in self.Ps:
            # Neural network evaluation
            state_tensor = state.get_state_tensor().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                log_ps, v = self.net(state_tensor)
                self.Ps[s] = torch.exp(log_ps).cpu().numpy()[0]
                v = v.item()

            valid_moves = state.get_valid_moves()
            valid_moves_mask = np.zeros(BOARD_COLS)
            valid_moves_mask[valid_moves] = 1
            self.Ps[s] = self.Ps[s] * valid_moves_mask
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # If all valid moves were masked make all valid moves equally probable
                self.Ps[s] = valid_moves_mask / np.sum(valid_moves_mask)

            self.Vs[s] = valid_moves
            self.Ns[s] = 0
            return -v

        cur_best = -float('inf')
        best_act = -1

        for a in self.Vs[s]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_state = state.copy()
        next_state.make_move(a)

        v = self.search(next_state)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def _get_state_key(self, state):
        return state.board.tobytes() + bytes([state.current_player])

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
                    (col * self.cell_size + self.cell_size // 2,
                     (row + 1) * self.cell_size + self.cell_size // 2),
                    self.cell_size // 2 - 5
                )

        pygame.display.update()

    def get_column_from_mouse(self, pos):
        x = pos[0]
        return x // self.cell_size

class Connect4Dataset(Dataset):
    def __init__(self, examples):
        self.states = [x[0] for x in examples]
        self.pis = [x[1] for x in examples]
        self.vs = [x[2] for x in examples]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.pis[idx], self.vs[idx]

def self_play(net, num_games=100, mcts_simulations=800):
    examples = []
    for i in range(num_games):
        mcts = MCTS(net, num_simulations=mcts_simulations)
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

        print(f'Self-play game {i + 1}/{num_games} completed')

    return examples

def train_network(net, examples, num_epochs=10, batch_size=256, lr=0.001, writer=None, min_loss=None):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scaler = GradScaler()  # For mixed precision training
    pi_criterion = nn.KLDivLoss(reduction='batchmean')
    v_criterion = nn.MSELoss()

    local_writer = writer
    if local_writer is None:
        local_writer = SummaryWriter('runs/connect4_training', flush_secs=1)

    global_step = 0

    # Calculate optimal number of workers for DataLoader
    num_workers = min(8, os.cpu_count() or 1)

    dataset = Connect4Dataset(examples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)

    for epoch in range(num_epochs):
        net.train()
        total_pi_loss = 0
        total_v_loss = 0
        batch_count = 0

        for state_batch, pi_batch, v_batch in data_loader:
            # Move tensors to the device
            state_batch = state_batch.to(DEVICE, non_blocking=True)
            pi_batch = pi_batch.to(DEVICE, non_blocking=True)
            v_batch = v_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Use autocast for mixed precision training
            with autocast():
                out_log_pi, out_v = net(state_batch)
                pi_loss = pi_criterion(out_log_pi, pi_batch)
                v_loss = v_criterion(out_v.squeeze(), v_batch.squeeze())
                total_loss = pi_loss + v_loss

            # Scale loss and backprop
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log metrics
            local_writer.add_scalar('Loss/Policy/Step', pi_loss.item(), global_step)
            local_writer.add_scalar('Loss/Value/Step', v_loss.item(), global_step)
            local_writer.add_scalar('Loss/Total/Step', total_loss.item(), global_step)

            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()
            batch_count += 1
            global_step += 1

            # Optional: Print progress every N batches
            if batch_count % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_count}, Loss: {total_loss.item():.4f}')

        # Log epoch metrics
        avg_pi_loss = total_pi_loss / batch_count
        avg_v_loss = total_v_loss / batch_count
        avg_total_loss = avg_pi_loss + avg_v_loss

        local_writer.add_scalar('Loss/Policy/Epoch', avg_pi_loss, epoch)
        local_writer.add_scalar('Loss/Value/Epoch', avg_v_loss, epoch)
        local_writer.add_scalar('Loss/Total/Epoch', avg_total_loss, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Average policy loss: {avg_pi_loss:.4f}')
        print(f'Average value loss: {avg_v_loss:.4f}')
        print(f'Average total loss: {avg_total_loss:.4f}')

        # Early stopping
        if min_loss is not None and avg_total_loss <= min_loss:
            print(f"Stopping training early at epoch {epoch + 1} because average total loss {avg_total_loss:.4f} "
                  f"reached the threshold {min_loss}")
            break

        # Force CUDA synchronization periodically
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    local_writer.flush()

def train_model(net, num_iterations=100, num_episodes=100, mcts_simulations=800, min_loss=None):
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
        train_network(net, examples, writer=writer, min_loss=min_loss)

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

        # Play AI vs AI to observe improvement
        print("Playing AI vs AI game to observe improvement")
        ai1 = AIPlayer(net, temperature=0.1)
        ai2 = AIPlayer(net, temperature=0.1)
        gui = Connect4GUI()
        result = play_game(ai1, ai2, gui, delay=500)
        pygame.quit()

        # Force synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    writer.close()

class AIPlayer:
    def __init__(self, net, mcts_simulations=800, temperature=0.1):
        self.net = net
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature

    def get_move(self, state):
        mcts = MCTS(self.net, num_simulations=self.mcts_simulations)
        pi = mcts.get_action_prob(state, temp=self.temperature)
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
    # Initialize network
    net = Connect4Net().to(DEVICE)

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Check for existing checkpoints
    try:
        checkpoint = torch.load('checkpoints/connect4_model_latest.pt', map_location=DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded existing model")
    except FileNotFoundError:
        print("Starting with fresh model")

    while True:
        print("\nConnect 4 Menu:")
        print("1. Train Model")
        print("2. Play Against AI")
        print("3. Watch AI vs AI")
        print("4. Watch AI vs AI from Checkpoint")
        print("5. Test TensorBoard")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            num_iterations = int(input("Enter number of iterations (default 10): ") or "10")
            num_episodes = int(input("Enter number of episodes per iteration (default 20): ") or "20")
            min_loss = float(input("Enter minimum loss for early stopping (default None): ") or "0.05")
            train_model(net, num_iterations=num_iterations, num_episodes=num_episodes, min_loss=min_loss)

        elif choice == '2':
            gui = Connect4GUI()
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
            elif result == 1:
                print("Player 1 (Red) wins!")
            elif result == 2:
                print("Player 2 (Yellow) wins!")
            else:
                print("Game was quit.")

        elif choice == '3':
            gui = Connect4GUI()
            ai1 = AIPlayer(net, temperature=0.1)
            ai2 = AIPlayer(net, temperature=0.1)
            result = play_game(ai1, ai2, gui, delay=500)

            if result == 0:
                print("Game ended in a draw!")
            elif result == 1:
                print("AI 1 (Red) wins!")
            elif result == 2:
                print("AI 2 (Yellow) wins!")
            else:
                print("Game was quit.")

        elif choice == '4':
            iteration = input("Enter iteration number to load (e.g., 0, 1, 2...): ")
            checkpoint_path = f'checkpoints/connect4_model_iter_{iteration}.pt'
            if iteration == "latest":
                checkpoint_path=checkpoint_path.replace("iter_", "")
                print(checkpoint_path)
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                net.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from iteration {iteration}")
                gui = Connect4GUI()
                ai1 = AIPlayer(net, temperature=0.1)
                ai2 = AIPlayer(net, temperature=0.1)
                result = play_game(ai1, ai2, gui, delay=500)
                if result == 0:
                    print("Game ended in a draw!")
                elif result == 1:
                    print("AI 1 (Red) wins!")
                elif result == 2:
                    print("AI 2 (Yellow) wins!")
                else:
                    print("Game was quit.")
            else:
                print(f"No checkpoint found for iteration {iteration}")

        elif choice == '5':
            test_tensorboard()

        elif choice == '6':
            break

        else:
            print("Invalid choice. Please try again.")

    pygame.quit()

if __name__ == "__main__":
    main()
