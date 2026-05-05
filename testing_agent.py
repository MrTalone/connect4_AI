import numpy as np
import pygame
import sys
import math  
from connect4_agent import Connect4Agent
MODEL_TRAIN_NAME = "Talon_Connect4.weights.h5"
MODEL_FINAL_NAME = "Talon_Connect4_Final.weights.h5"

# ---------- Configuration ----------
TRAINMODE = True
SMART_MOVE = False #make oppnent run off an ai selection over random
PLAYER = False #allow a player to play the game

##
RENDER = True
RENDER_EACH_MOVE = False
RENDER_TEXT = False
CONNECT = 4

# ---------- Constants for drawing ----------
#board setup
ROWS = 6
COLUMNS = 7
PLAYER_PIECE = '+'
MODEL_PIECE = '-'

#colors
BLUE = (0, 0, 255)
MODEL_GRAPHIC = (244, 179, 0) 
PLAYER_GRAPHIC = (255, 0, 0)

#game size
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE / 2 - 6)
width = COLUMNS * SQUARE_SIZE
height = (ROWS + 1) * SQUARE_SIZE

# ---------- Helper functions ----------
def random_move(env):
    """Choose a random valid move from the current board."""
    valid = [c for c in range(COLUMNS) if env.is_valid_move(c)]
    if not valid:
        return 0  # fallback, should never happen
    return np.random.choice(valid)

#from pong ai homework
def discount_rewards(r, gamma=0.99):
    #Change Gamma to play with discounting
    #Yes its GAMMA lowercase!
    r = np.array(r, dtype=np.float32)
    discounted = np.zeros_like(r)
    running = 0.0
    for t in reversed(range(len(r))):
        running = running * gamma + r[t]
        discounted[t] = running
    std = discounted.std()
    if std > 1e-8:
        discounted = (discounted - discounted.mean()) / std
    return discounted


def draw_board(board):
    """Render the Connect4 board using Pygame."""
    if not RENDER:
        return
    screen.fill((0, 0, 0))
    # Draw grid and holes
    for r in range(ROWS):
        for c in range(COLUMNS):
            pygame.draw.rect(
                screen,
                BLUE,
                (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE,SQUARE_SIZE, SQUARE_SIZE)
            )
            pygame.draw.circle(
                screen,
                (0, 0, 0),
                (
                    int(c * SQUARE_SIZE + SQUARE_SIZE/2),
                    int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE/2)
                ),
                int(RADIUS)
            )
    # Draw pieces
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r][c] == PLAYER_PIECE:
                color = PLAYER_GRAPHIC     # red
            elif board[r][c] == MODEL_PIECE:
                color = MODEL_GRAPHIC   # gold
            else:
                continue
            pygame.draw.circle(screen, color,
                               (int(c * SQUARE_SIZE + SQUARE_SIZE/2),
                                height - int(r * SQUARE_SIZE + SQUARE_SIZE/2)),
                               RADIUS)
    pygame.display.update()

def getPosition(pos):
    posx = pos[0]
    return int(math.floor(posx/SQUARE_SIZE))

# def isValidInput(board,col):
#     return board[(ROWS-1)][col] == ' '



def printBoard(board):
    print(np.flip(board,0))
    print("-" * 20)
    pygame.time.wait(100) 
    #printing the text based game out here

# ---------- Environment (with winner tracking) ----------
class Connect4Env:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLUMNS
        self.score = [0, 0]
        self.done = False
        self.winner = None
        self.reset()

    def reset(self):
        self.board = np.full((ROWS, COLUMNS), ' ')
        self.current_player = np.random.randint(2)   # 0 = PLAYER_PIECE, 1 = MODEL_PIECE (agent)
        self.done = False
        self.winner = None          # '+' or '-' or None
        return self.get_state()

    #makes sense to be in the environment sense im not necisarrly looking at frams at the moment
    def get_state(self):
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == PLAYER_PIECE:
                    state[0, r, c] = 1.0
                elif self.board[r][c] == MODEL_PIECE:
                    state[1, r, c] = 1.0
        state[2, :, :] = self.current_player
        return state.ravel()

    def is_valid_move(self, col):
        return self.board[self.rows - 1][col] == ' '

    def get_open_row(self, col):
        for r in range(self.rows):
            if self.board[r][col] == ' ':
                return r
        return -1

    def step(self, action):
        reward = 0.0
        done = False
        winner = None

        # if PLAYER:
        #     print("hello player")
        #     #allower user to play
        
        
        if self.done:
            return self.get_state(), 0, True, self.winner

        if not self.is_valid_move(action):
            self.done = True
            self.winner = None
            return self.get_state(), -1.0, True, None

        row = self.get_open_row(action)
        piece = PLAYER_PIECE if self.current_player == 0 else MODEL_PIECE
        self.board[row][action] = piece

        
        if self.check_win(piece):
            self.done = True
            self.winner = piece
            if piece == MODEL_PIECE:
                #print("MODEL WIN")
                self.score[1] += 1
                reward = 1.0
            else:
                #print("PLAYER WIN")
                self.score[0] += 1
                reward = -1.0
        elif self._is_draw():
            self.done = True
            self.winner = None
            reward = 0.5
        else:
            self.current_player = 1 - self.current_player
            reward = 0.0

        return self.get_state(), reward, self.done,self.winner

    def check_win(self,piece):
        #horizontal
        board = self.board
        for r in range(ROWS):
            for c in range(COLUMNS-CONNECT+1):
                if all(board[r][c+i] == piece for i in range(CONNECT)):
                    return True
        
        #vertical
        for c in range(COLUMNS):
            for r in range(ROWS-CONNECT+1):
                if all(board[r+i][c] == piece for i in range(CONNECT)):
                    return True

        #diag up right
        for r in range(ROWS-CONNECT+1):
            for c in range(COLUMNS-CONNECT+1):
                if all(board[r+i][c+i] == piece for i in range(CONNECT)):
                    return True
        
        #diag down right
        for r in range(CONNECT-1,ROWS):
            for c in range(COLUMNS-CONNECT+1):
                if all(board[r-i][c+i] == piece for i in range(CONNECT)):
                    return True

        return False

        #this worked for basic connect 4 but i wanted some better fucntionality
        # #check horizontal
        # for c in range(COLUMNS-3):
        #     for r in range(ROWS):
        #         if board[r][c] == piece and board[r][c+1]==piece and board[r][c+2]==piece and board[r][c+3]==piece:
        #             return True
                
        
        # #check vertical locations
        # for c in range(COLUMNS):
        #     for r in range(ROWS-3):
        #         if board[r][c] == piece and board[r+1][c] ==piece and board[r+2][c] ==piece and board[r+3][c] ==piece:
        #             return True

        # #check pos slope
        # for c in range(COLUMNS-3):
        #     for r in range(ROWS-3):
        #         if board[r][c] == piece and board[r+1][c+1]==piece and board[r+2][c+2]==piece and board[r+3][c+3]==piece:
        #             return True

        # #check neg slope
        # for c in range(COLUMNS-3):
        #     for r in range(3,ROWS):
        #         if board[r][c] == piece and board[r-1][c+1]==piece and board[r-2][c+2]==piece and board[r-3][c+3]==piece:
        #             return True

    def _is_draw(self):
        return np.all(self.board != ' ')

# ---------- Main (your structure, fixed) ----------
if __name__ == "__main__":
    env = Connect4Env()


    if PLAYER:
        #TRAINMODE = False
        RENDER = True
        RENDER_EACH_MOVE = True
        agent = Connect4Agent(MODEL_FINAL_NAME)
        print("HUMAN VS AI MODE")
    else:
        agent = Connect4Agent(MODEL_TRAIN_NAME)


    if RENDER:
        pygame.init()
        SQUARE_SIZE = 100
        RADIUS = int(SQUARE_SIZE/2 - 6)
        width = COLUMNS * SQUARE_SIZE
        height = (ROWS + 1) * SQUARE_SIZE
        screen = pygame.display.set_mode((width, height))
        draw_board(env.board)
        pygame.display.update()
        clock = pygame.time.Clock()   # good to control frame rate

    else:
        print("Running without display mode")

    episodes = 0
    point_results = []      # 1 = agent win, 0 = loss/draw

    try:
        while True:
            episodes += 1
            state = env.reset()
            done = False

            #render_this = RENDER and (episodes % RENDER_EVERY == 0)

            # Clear episode data (do NOT accumulate across episodes)
            if TRAINMODE:
                states, actions, rewards = [], [], []

            # ---------- GAME LOOP (inner) ----------
            while not done:
                # Handle quit event if rendering
                if RENDER:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                # Get valid moves from current board
                valid_mask = [env.is_valid_move(c) for c in range(COLUMNS)]
                valid_indices = [c for c, ok in enumerate(valid_mask) if ok]
                # Choose action based on whose turn it is
                if env.current_player == 1:      # Agent's turn
                    action = agent.get_action(state, valid_mask)
                else:                            # Player (random for training)
                    if PLAYER:
                        action = None
                        draw_board(env.board)
                        pygame.display.update()

                        #wait for player
                        while action is None:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()

                                if event.type == pygame.MOUSEMOTION:
                                    draw_board(env.board)

                                    # Clear the top bar and draw the hovering piece
                                    #pygame.draw.rect(screen, (0, 0, 0), (0, 0, width, SQUARE_SIZE))
                                    col = getPosition(event.pos)
                                
                                    if 0 <= col < COLUMNS:
                                        pygame.draw.circle(screen, PLAYER_GRAPHIC,
                                                        (event.pos[0], int(SQUARE_SIZE / 2)),
                                                        RADIUS)
                                    pygame.display.update()

                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    col = getPosition(event.pos)

                                    print(f"SELECTED COL: {col}")

                                    if col in valid_indices:
                                        action = col
                                    else:
                                        print("ERROR: invalid move, try again")
                     
                            clock.tick(30)
                    elif SMART_MOVE:
                        action = agent.get_action(state, valid_mask)
                    else:
                        action = random_move(env)

                # Take a step in the environment
                next_state, reward, done, winner = env.step(action)
                if RENDER_EACH_MOVE:
                    draw_board(env.board)          # draw the final board (winning piece visible)
                    pygame.display.update()
                    pygame.time.wait(50)

                # Store transition for training (state BEFORE action)
                if TRAINMODE:
                    states.append(state)          # important: state, not next_state
                    actions.append(int(action))
                    rewards.append(float(reward))

                # Move to next state
                state = next_state
                if RENDER_TEXT:
                    printBoard(env.board)

            # ---------- EPISODE FINISHED ----------
            # ---------- AFTER GAME ENDS: SHOW FINAL BOARD ----------
            if RENDER:
                draw_board(env.board)          # draw the final board (winning piece visible)
                pygame.display.update()
                pygame.time.wait(1500)         # pause to see the result
            if RENDER_TEXT:
                printBoard(env.board)

            if winner == MODEL_PIECE:
                point_results.append(1)      # agent win
            else:
                point_results.append(0)      # loss or draw

            # Train agent on the episode data
            if TRAINMODE and len(states) > 0:
                discounted = discount_rewards(rewards)
                agent.training_step(states, actions, discounted)

            # ---------- LOGGING every 10 episodes (asked chat for some print) ----------
            if episodes % 10 == 0:
                recent = point_results[-10:] if len(point_results) >= 10 else point_results
                win_rate = np.mean(recent) * 100
                total_wins = sum(point_results)
                print(f"Episode {episodes} | Score CPU={env.score[0]} Agent={env.score[1]} | "
                      f"Win% (last 10): {win_rate:.1f}% | Total wins: {total_wins}")

            # ---------- SAVE CHECKPOINT every 100 episodes ----------
            if episodes % 100 == 0:
                SMART_MOVE = not SMART_MOVE
                win_rate_100 = np.mean(point_results[-100:]) * 100 if len(point_results) >= 100 else 0
                print(f"Points: {len(point_results)} | Score CPU={env.score[0]} Agent={env.score[1]} | "
                      f"Win% (last 100): {win_rate_100:.1f}%")
                agent.save_model("Talon_Connect4.weights.h5")
                print("Checkpoint saved.")

    except KeyboardInterrupt:
        print("\nSaving final model and exiting...")
        agent.save_model("Talon_Connect4_final.weights.h5")
        if RENDER:
            pygame.quit()
        sys.exit()