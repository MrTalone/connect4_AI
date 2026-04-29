"""
Connect 4 with Reinforcement Learning (DQN)
- Train an AI by self-play
- Play against the trained AI (graphics or console)
"""

import numpy as np
import pygame
import sys
import math
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time


# -------------------------------
# 1. Game Logic (original)
# -------------------------------
ROWS = 6
COLUMNS = 7
PLAYER_PIECE = '+'
MODEL_PIECE = '-'
CONNECT = 4

def create_board():
    return np.full((ROWS, COLUMNS), ' ')

def dropPiece(board, row, col, piece):
    board[row][col] = piece

def isValidInput(board, col):
    return board[ROWS-1][col] == ' '

def getOpenRow(board, col):
    for r in range(ROWS):
        if board[r][col] == ' ':
            return r
    return -1

def printBoard(board):
    print(np.flip(board, 0))

def checkWin(board, piece):

    # horizontal
    for r in range(ROWS):
        for c in range(COLUMNS - CONNECT + 1):
            if all(board[r][c+i] == piece for i in range(CONNECT)):
                return True
    # vertical
    for c in range(COLUMNS):
        for r in range(ROWS - CONNECT + 1):
            if all(board[r+i][c] == piece for i in range(CONNECT)):
                return True
    # diagonal up right
    for r in range(ROWS - CONNECT + 1):
        for c in range(COLUMNS - CONNECT + 1):
            if all(board[r+i][c+i] == piece for i in range(CONNECT)):
                return True
    # diagonal down right
    for r in range(CONNECT-1, ROWS):
        for c in range(COLUMNS - CONNECT + 1):
            if all(board[r-i][c+i] == piece for i in range(CONNECT)):
                return True
    return False

def isDraw(board):
    return np.all(board != ' ')

# -------------------------------
# 2. RL Environment
# -------------------------------
class Connect4Env:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLUMNS
        self.action_space = COLUMNS
        self.reset()

    def reset(self):
        self.board = create_board()
        self.current_player = 0   # 0 = PLAYER_PIECE, 1 = MODEL_PIECE
        self.done = False
        return self._get_state()

    def _get_state(self):
        """3-channel state: (current player pieces, opponent pieces, current player indicator)"""
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == self._piece_of_player(self.current_player):
                    state[0, r, c] = 1.0
                elif self.board[r][c] == self._piece_of_player(1 - self.current_player):
                    state[1, r, c] = 1.0
        # Channel 2: all ones to indicate whose turn it is
        state[2, :, :] = 1.0
        return state

    def _piece_of_player(self, player):
        return PLAYER_PIECE if player == 0 else MODEL_PIECE

    def is_valid_move(self, col):
        return 0 <= col < self.cols and self.board[self.rows-1][col] == ' '

    def get_open_row(self, col):
        return getOpenRow(self.board, col)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        if not self.is_valid_move(action):
            # invalid move: end game with penalty
            return self._get_state(), -0.5, True, {"invalid": True}

        row = self.get_open_row(action)
        piece = self._piece_of_player(self.current_player)
        dropPiece(self.board, row, action, piece)

        if checkWin(self.board, piece):
            self.done = True
            reward = 1.0
        elif isDraw(self.board):
            self.done = True
            reward = 0.0
        else:
            # switch player
            self.current_player = 1 - self.current_player
            reward = 0.0   # optional: -0.01 for longer games

        return self._get_state(), reward, self.done, {}

    def render(self, mode='human'):
        """Optional console render"""
        if mode == 'human':
            print(np.flip(self.board, 0))

# -------------------------------
# 3. DQN Model and Agent
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After convolutions, shape: (128, ROWS, COLS)
        self.fc = nn.Sequential(
            nn.Linear(128 * ROWS * COLUMNS, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.9995, batch_size=64, target_update=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy_net = DQN((3, ROWS, COLUMNS), env.action_space).to(self.device)
        self.target_net = DQN((3, ROWS, COLUMNS), env.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def act(self, state, eval_mode=False):
        """Return action (epsilon-greedy). If eval_mode, epsilon=0."""
        eps = 0.0 if eval_mode else self.epsilon
        if np.random.random() < eps:
            # choose random valid action
            valid = [col for col in range(self.env.action_space) if self.env.is_valid_move(col)]
            return random.choice(valid) if valid else 0
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t).cpu().numpy()[0]
            # mask invalid moves
            for col in range(self.env.action_space):
                if not self.env.is_valid_move(col):
                    q_values[col] = -np.inf
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1, keepdim=True)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="connect4_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)

    #def load(self, path="connect4_dqn.pth"):


# -------------------------------
def train(episodes=20000, render_every=5, save_every=5000):
    env = Connect4Env()
    agent = DQNAgent(env)
    scores = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        move_count = 0
        #last_action = None
        while not done:
            if episode % render_every == 0:
                #                env.render(last_move=last_action)
                env.render()

                time.sleep(0.1)


            

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            #last_action = action
            agent.replay()
            agent.steps += 1
            move_count += 1
            if agent.steps % agent.target_update == 0:
                agent.update_target()

        scores.append(total_reward)
        if episode % 10 == 0:   # print every 10 episodes
            # env.render()
            # time.sleep(0.1)   # now safe
            print(f"Episode {episode} finished in {move_count} moves, avg reward: {np.mean(scores[-100:]):.2f}")
        if episode % render_every == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Avg Reward (last 100): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, Moves: {move_count}")
        if episode % save_every == 0 and episode > 0:
            agent.save(f"connect4_dqn_ep{episode}.pth")

    agent.save("connect4_dqn_final.pth")
    print("Training finished. Model saved.")
    return agent

# -------------------------------
# 5. Human vs AI (with pygame or console)
# -------------------------------
def play_vs_ai(use_graphics=True, model_path="connect4_dqn_final.pth"):
    env = Connect4Env()
    agent = DQNAgent(env)
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("No trained model found. Training a fresh one (this will take a while).")
        agent = train(episodes=10000)

    board = create_board()
    game_over = False
    turn = 0  # 0 = human (PLAYER_PIECE), 1 = AI (MODEL_PIECE)

    if use_graphics:
        pygame.init()
        SQUARE_SIZE = 100
        RADIUS = int(SQUARE_SIZE/2 - 6)
        width = COLUMNS * SQUARE_SIZE
        height = (ROWS + 1) * SQUARE_SIZE
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Connect 4 - Play vs AI")
        font = pygame.font.SysFont("monospace", 75)

        def draw_board_gui(board):
            screen.fill((0,0,0))
            for r in range(ROWS):
                for c in range(COLUMNS):
                    pygame.draw.rect(screen, (0,0,255), (c*SQUARE_SIZE, r*SQUARE_SIZE+SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    pygame.draw.circle(screen, (0,0,0), (int(c*SQUARE_SIZE+SQUARE_SIZE/2), int(r*SQUARE_SIZE+SQUARE_SIZE+SQUARE_SIZE/2)), RADIUS)
            for r in range(ROWS):
                for c in range(COLUMNS):
                    if board[r][c] == PLAYER_PIECE:
                        pygame.draw.circle(screen, (255,0,0), (int(c*SQUARE_SIZE+SQUARE_SIZE/2), height - int(r*SQUARE_SIZE+SQUARE_SIZE/2)), RADIUS)
                    elif board[r][c] == MODEL_PIECE:
                        pygame.draw.circle(screen, (0,200,100), (int(c*SQUARE_SIZE+SQUARE_SIZE/2), height - int(r*SQUARE_SIZE+SQUARE_SIZE/2)), RADIUS)
            pygame.display.update()

        draw_board_gui(board)

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if turn == 0:  # human turn
                    if event.type == pygame.MOUSEMOTION:
                        pygame.draw.rect(screen, (0,0,0), (0,0,width,SQUARE_SIZE))
                        posx = event.pos[0]
                        pygame.draw.circle(screen, (255,0,0), (posx, int(SQUARE_SIZE/2)), RADIUS)
                        pygame.display.update()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        posx = event.pos[0]
                        col = int(math.floor(posx / SQUARE_SIZE))
                        if isValidInput(board, col):
                            row = getOpenRow(board, col)
                            dropPiece(board, row, col, PLAYER_PIECE)
                            if checkWin(board, PLAYER_PIECE):
                                print("You win!")
                                game_over = True
                            elif isDraw(board):
                                print("Draw!")
                                game_over = True
                            turn = 1
                            draw_board_gui(board)
                else:  # AI turn
                    # Convert current board to agent's state (current_player = 1)
                    # Temporarily set env.board and current_player to match
                    env.board = board.copy()
                    env.current_player = 1
                    state = env._get_state()
                    action = agent.act(state, eval_mode=True)
                    if isValidInput(board, action):
                        row = getOpenRow(board, action)
                        dropPiece(board, row, action, MODEL_PIECE)
                        if checkWin(board, MODEL_PIECE):
                            print("AI wins!")
                            game_over = True
                        elif isDraw(board):
                            print("Draw!")
                            game_over = True
                        turn = 0
                        draw_board_gui(board)
                        pygame.time.wait(300)

            if game_over:
                pygame.time.wait(2000)
                pygame.quit()
                sys.exit()

    else:
        # console version
        while not game_over:
            printBoard(board)
            if turn == 0:
                col = int(input(f"Your turn (0-{COLUMNS-1}): "))
                if isValidInput(board, col):
                    row = getOpenRow(board, col)
                    dropPiece(board, row, col, PLAYER_PIECE)
                    if checkWin(board, PLAYER_PIECE):
                        print("You win!")
                        game_over = True
                    elif isDraw(board):
                        print("Draw!")
                        game_over = True
                    turn = 1
            else:
                env.board = board.copy()
                env.current_player = 1
                state = env._get_state()
                col = agent.act(state, eval_mode=True)
                print(f"AI chooses column {col}")
                if isValidInput(board, col):
                    row = getOpenRow(board, col)
                    dropPiece(board, row, col, MODEL_PIECE)
                    if checkWin(board, MODEL_PIECE):
                        print("AI wins!")
                        game_over = True
                    elif isDraw(board):
                        print("Draw!")
                        game_over = True
                    turn = 0

# -------------------------------
# 6. Main entry point
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="play")
    parser.add_argument("--graphics", action="store_true", help="Use pygame graphics when playing")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    args = parser.parse_args()

    if args.mode == "train":
        train(episodes=args.episodes)
    else:
        play_vs_ai(use_graphics=args.graphics)