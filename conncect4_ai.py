#TALON FINEHOUT

#this connect 4 game follows some setup from keith Galli on youtube
#How to Program Connect 4 in Python!(1-4)
#https://www.youtube.com/playlist?list=PLFCB5Dp81iNV_inzM-R9AKkZZlePCZdtV

import numpy as np
import pygame
import sys
import math
from connect4_agent import Connect4Agent


TRAINMODE = True
RENDER = True

ROWS = 6
COLUMNS=7
PLAYER_PIECE = '+'
MODEL_PIECE = '-'
CONNECT = 4
BLUE=(0,0,255)


gameOver = False
userTurn = 0

#--------------- REGULARE TEXT BASED AND GAME FUNCTION ---------------
def create_board():
    board= np.full((ROWS,COLUMNS),' ')
    return board

board = create_board()
print(board)

def dropPiece(board, row, col, piece):
    board[row][col] = piece

def isValidInput(board,col):
    return board[(ROWS-1)][col] == ' '

def getOpenRow(board,col):
    for r in range(ROWS):
        if board [r][col]==' ':
            return r
    return -1

def printBoard(board):
    print(np.flip(board,0))

# -------------------------- GRAPHICS ------------------------------
def draw_board(board):
    for r in range(ROWS):
        for c in range(COLUMNS):
            pygame.draw.rect(
                screen,
                BLUE,
                (c * SQUARE_SIZE, r * SQUARE_SIZE+SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )
            pygame.draw.circle(
                screen,
                (0,0,0),
                (
                    int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                    int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)
                ),
                int(RADIUS)
            )


    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r][c]=='+':
                pygame.draw.circle(
                    screen,
                    (255,0,0),
                    (
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        height- int(r * SQUARE_SIZE  + SQUARE_SIZE / 2)
                    ),
                    int(RADIUS)
                )
            elif board[r][c]=='-':
                pygame.draw.circle(
                    screen,
                    (0,200,100),
                    (
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        height- int(r * SQUARE_SIZE + SQUARE_SIZE / 2)
                    ),
                    int(RADIUS)
                )


def checkWin(board,piece):
    #horizontal
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

def isDraw(board):
    return np.all(board != ' ')


## SET UP RL AGENT ##

class Connect4Env:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLUMNS
        self.action = COLUMNS
        self.reset()

    def reset(self):
        self.board = create_board()
        self.current_player = 0 #0 player, 1 agent
        self.done = False
        return self.get_state()

#
    def get_state(self):
        #current player, opponent/model
        state = np.zeros((3,self.rows,self.cols),dtype = np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == self.piece_of_player(self.current_player):
                    state[0, r, c] = 1.0
                elif self.board[r][c] == self.piece_of_player(1 - self.current_player):
                    state[1, r, c] = 1.0
        state[2,:,:] = 1.0 if self.current_player == 0 else 0.0        
        return state.ravel()

    def piece_of_player(self,player):
        if player == 0:
            return PLAYER_PIECE
        else:
            return MODEL_PIECE
        

    def is_valid_move(self,col):
        return self.board[(self.rows-1)][col] == ' '

    def get_open_row(self,col):
        return getOpenRow(self.board,col)

    def step(self,action):
        #(state,reward,done)
        if self.done:
            return self.get_state(), 0 ,True
        
        if not self.is_valid_move(action):
            return self.get_state(),-1,True # dont make bad moves

        #let the agent find where to place a peice and place it
        row = self.get_open_row(action)
        piece = self.piece_of_player(self.current_player)
        dropPiece(self.board,row,action,piece)

        #check if the piece led to a win or is a draw, other wise let the ohter player play
        if checkWin(self.board,piece):
            self.done = True
            reward = 1.0
        elif isDraw(self.board):
            self.done = True
            reward = 0.0
        else:
            #neither happened - game is still going
            self.current_player = 1 - self.current_player
            reward = 0.0

        return self.get_state(),reward,self.done
    

    def render(self):
        print(np.flip(self.board,0))

#from Pong RL Homeworkd cs421
def discount_rewards(r,gamma=0.99):
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

#connect 4 is turned bases
#reward winner at end of round

#start env

#make training loop
if __name__ == "__main__":
    env = Connect4Env()
    agent = Connect4Agent()

    episodes = 0

    if RENDER:
        pygame.init()

        SQUARE_SIZE = 100
        RADIUS = int(SQUARE_SIZE/2-6)
        width = COLUMNS*SQUARE_SIZE
        height = (ROWS+1)*SQUARE_SIZE
        screenSize = (width,height)
        screen = pygame.display.set_mode(screenSize)

        draw_board(board)
        pygame.display.update()

    while True:
        episodes += 1

        state = env.reset()
        done = False

        states, actions, rewards = [], [], []

        while not done:

            if RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            # get valid moves
            valid_moves = [c for c in range(COLUMNS) if env.is_valid_move(c)]

            action = agent.get_action(state, valid_moves)

            next_state, reward, done = env.step(action)

            # store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        if RENDER:
            draw_board(env.board)
            pygame.time.wait(50) 

        # AFTER episode ends
        total_reward = sum(rewards)

        
        # discount rewards (for future learning step)
        discounted = discount_rewards(rewards)
        loss=agent.train_step(states,actions,discounted)


        if episodes % 10 == 0:
            print(f"Episode {episodes} | Reward: {total_reward}")

        if episodes % 100 == 0:
            agent.save_model()





