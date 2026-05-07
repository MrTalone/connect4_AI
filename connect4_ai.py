#TALON FINEHOUT

#this connect 4 game follows some setup from keith Galli on youtube
#How to Program Connect 4 in Python!(1-4)
#https://www.youtube.com/playlist?list=PLFCB5Dp81iNV_inzM-R9AKkZZlePCZdtV
#then used a lot of CS-421 png agent code to buil around the game with env

import numpy as np
import pygame
import sys
import math
#import agents from connect4_agent.py
from connect4_agent import Connect4Agent as Connect4Agent_trainer
from connect4_agent import Connect4Agent as Connect4Agent_self_play

MODEL_TRAIN_NAME = "Talon_Connect4.weights.h5"
MODEL_FINAL_NAME = "Talon_Connect4_Final.weights.h5"


#---------- CONFIGS -----------

#set if training, user is playing, or if cpu is playing from agent or random
TRAINMODE = True
SMART_MOVE = False #make oppnent run off an ai selection over random
PLAYER =  False#allow a player to play the game

#renders - note will be force to true if player is true
RENDER = True
RENDER_EACH_MOVE = False
RENDER_TEXT = False

#board settings
ROWS = 6
COLUMNS=7
MODEL_PLAYER_ID = 1
PLAYER_PIECE = '+'
MODEL_PIECE = '-'
CONNECT = 4

#render colors
BLUE=(0,0,255)
BLUE = (0, 0, 255)
MODEL_GRAPHIC = (244, 179, 0) 
PLAYER_GRAPHIC = (255, 0, 0)

#render size
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE / 2 - 6)
width = COLUMNS * SQUARE_SIZE
height = (ROWS + 1) * SQUARE_SIZE

#NEW BATCH SIZE LEARNING - will it work??
BATCH_SIZE = 1000
# gameOver = False
# userTurn = 0

#-------------- helper functions to use later --------------
#choose a random move that is valid (open collumn)
# def isValidInput(board,col):
#     return board[(ROWS-1)][col] == ' '
def random_move(env):
    valid = [c for c in range(COLUMNS) if env.is_valid_move(c)]
    if valid:
        return np.random.choice(valid)
    else:
        return 0
    
    

#discount rewards (from pong game)
def discount_rewards(r, gamma=0.90):
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


#--------------- REGULARE TEXT BASED AND GAME FUNCTION (OLD) ---------------
#SHOLD NOT NEED THIS ANYMORE
# def create_board():
#     board= np.full((ROWS,COLUMNS),' ')
#     return board

# board = create_board()
# print(board)

# def dropPiece(board, row, col, piece):
#     board[row][col] = piece

# def isValidInput(board,col):
#     return board[(ROWS-1)][col] == ' '

# def getOpenRow(board,col):
#     for r in range(ROWS):
#         if board [r][col]==' ':
#             return r
#     return -1

##--------------still want to print text based game if needed--------------
def printBoard(board):
    print(np.flip(board,0))
    print("-" * 20)
    pygame.time.wait(100)#timer to wait for next placement

# -------------------------- GRAPHICS ------------------------------
def draw_board(board):
    if not RENDER:
        return
    screen.fill((0,0,0)) #for loading

    #grid
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

    #pieces
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r][c]== PLAYER_PIECE:
                pygame.draw.circle(
                    screen,
                    PLAYER_GRAPHIC,
                    (
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        height- int(r * SQUARE_SIZE  + SQUARE_SIZE / 2)
                    ),
                    int(RADIUS)
                )
            elif board[r][c]==MODEL_PIECE:
                pygame.draw.circle(
                    screen,
                    MODEL_GRAPHIC,
                    (
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        height- int(r * SQUARE_SIZE + SQUARE_SIZE / 2)
                    ),
                    int(RADIUS)
                )
    pygame.display.update()

#--------------when player true, need to get pos clicked and find best collumn--------------
def getPosition(pos):
    posx = pos[0]
    return int(math.floor(posx/SQUARE_SIZE))

#HANDLED IN ENV
# def checkWin(board,piece):
#     #horizontal
#     for r in range(ROWS):
#         for c in range(COLUMNS-CONNECT+1):
#             if all(board[r][c+i] == piece for i in range(CONNECT)):
#                 return True
    
#     #vertical
#     for c in range(COLUMNS):
#         for r in range(ROWS-CONNECT+1):
#             if all(board[r+i][c] == piece for i in range(CONNECT)):
#                 return True

#     #diag up right
#     for r in range(ROWS-CONNECT+1):
#         for c in range(COLUMNS-CONNECT+1):
#             if all(board[r+i][c+i] == piece for i in range(CONNECT)):
#                 return True
    
#     #diag down right
#     for r in range(CONNECT-1,ROWS):
#         for c in range(COLUMNS-CONNECT+1):
#             if all(board[r-i][c+i] == piece for i in range(CONNECT)):
#                 return True
#     return False

# def isDraw(board):
#     return np.all(board != ' ')


##---------------------------- SET UP RL AGENT---------------------------- ##

class Connect4Env:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLUMNS
        #self.action = COLUMNS
        self.score = [0,0]
        self.done = False
        self.winner = None
        self.reset()

    def reset(self):
        self.board = np.full((ROWS, COLUMNS), ' ')
        self.current_player = 0   # 0 = PLAYER_PIECE, 1 = MODEL_PIECE (agent) #randomply start with different players
        self.done = False
        self.winner=None
        return self.get_state()

    #makes sense to be in the environment sense im not necisarrly looking at frams at the moment

    def get_state(self):
        #current player, opponent/model
        #3 channels
        #   0 = player piece, 1 if filled, 0 otherwise
        #   1 = agent piece, 1 if filled, 0 otherwise
        #  2 = whose turn it is, 1 player, 0 agent (still a row*col type)
        #with addition row*collumn
        
        state = np.zeros((3,self.rows,self.cols),dtype = np.float32)
        for r in range(self.rows):
            for c in range(self.cols):

                if self.board[r][c] == PLAYER_PIECE:
                    state[0, r, c] = 1.0
                elif self.board[r][c] == MODEL_PIECE:
                    state[1, r, c] = 1.0
        state[2,:,:] = 1.0 if self.current_player == MODEL_PLAYER_ID else 0.0
        
        # # Height channel: fill from bottom with 1s up to the current height
        # for c in range(self.cols):
        #     height = 0
        #     for r in range(self.rows):
        #         if self.board[r][c] != ' ':
        #             height += 1
        #     # Set the bottom 'height' cells to 1.0 (normalised height not needed because it's already 0..ROWS)
        #     for r in range(height):
        #         state[3, r, c] = 1.0
        
        return state

    def piece_of_player(self,player):
        if player == 0:
            return PLAYER_PIECE
        else:
            return MODEL_PIECE
   
    def is_valid_move(self,col):
        return self.board[(self.rows-1)][col] == ' '

    def get_open_row(self,col):
        for r in range(self.rows):
            if self.board[r][col] == ' ':
                return r
        return -1
        #return getOpenRow(self.board,col)

    def step(self,action):
        reward = 0.0
        done = False
        winner = None

        # if PLAYER:
        #     print("hello player")
        #     #allower user to play
        

        #(state,reward,done)
        if self.done:
            return self.get_state(), 0 ,True, self.winner
        
        if not self.is_valid_move(action):
            self.done = True
            self.winner = None
            return self.get_state(), -1.0, True, None

        #let the agent find where to place a peice and place it
        row = self.get_open_row(action)
        piece = MODEL_PIECE if self.current_player == MODEL_PLAYER_ID else PLAYER_PIECE        
        self.board[row][action] = piece #place piece

        #piece = PLAYER_PIECE if self.current_player == 0 else MODEL_PIECE
        #dropPiece(self.board,row,action,piece)

        #---------------------------- REWARDS----------------------------
        #check for threats (if ai has connect 3, reward)
        #if player has connect 3, punish
        if piece == MODEL_PIECE:
            reward += 0.5 * self.count_threats(MODEL_PIECE, 3)
        elif piece == PLAYER_PIECE:
            reward -= 0.5 * self.count_threats(PLAYER_PIECE, 3)


        #------------------------------------------------------------------------------------
        #check if the piece led to a win or is a draw, other wise let the ohter player play
        #connect 4 is turned bases
        #reward winner at end of round 
        if self.checkWin(piece):
            self.done = True
            self.winner = piece

            #-------- MODEL WINNER #--------
            if piece == MODEL_PIECE:
                self.score[1]+=1
                reward += 5.0
            else:
                self.score[0] += 1
                reward += -5.0
        elif self.isDraw():
            self.done = True
            self.winner=None
            reward = 0.1#small reward for draw, better than loosing, worst than winning
        else:
            #neither happened - game is still going
            self.current_player = 1 - self.current_player
            reward = 0.0

        return self.get_state(), reward, self.done,self.winner
    
    def checkWin(self,piece):
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

    def isDraw(self):
        return np.all(self.board != ' ')

    #time to add some more rewards (got some help from chat for counting how many are next to each other)    
    #(asked chat for a threat count setup - purpose is to find how many oppent peices are next to eachother and try to block pot wins)
    #very similar to win check
    def count_threats(self, piece, length):
        """Return number of times 'piece' appears exactly 'length' times consecutively
        in any direction, and the line is not blocked by the same piece at either end."""
        board = self.board
        rows, cols = self.rows, self.cols
        threats = 0
        #if there is a section all as "PIECE" to a length
        #Returns True: If every element in the iterable evaluates to True, or if the iterable is empty.Returns False: If even one element evaluates to False. (https://www.w3schools.com/python/ref_func_all.asp#:~:text=Module%20Reference,in%20a%20dictionary%20are%20True:)
                
        # Horizontal
        for r in range(rows):
            for c in range(cols - length + 1):
                if all(board[r][c+i] == piece for i in range(length)):
                    if (c == 0 or board[r][c-1] != piece) and (c+length == cols or board[r][c+length] != piece):
                        threats += 1

        # Vertical
        for c in range(cols):
            for r in range(rows - length + 1):
                if all(board[r+i][c] == piece for i in range(length)):
                    if (r == 0 or board[r-1][c] != piece) and (r+length == rows or board[r+length][c] != piece):
                        threats += 1

        # Diagonal down-right (\)
        for r in range(rows - length + 1):
            for c in range(cols - length + 1):
                if all(board[r+i][c+i] == piece for i in range(length)):
                    if ((r == 0 or c == 0 or board[r-1][c-1] != piece) and
                        (r+length == rows or c+length == cols or board[r+length][c+length] != piece)):
                        threats += 1

        # Diagonal up-right (/)
        for r in range(length-1, rows):
            for c in range(cols - length + 1):
                if all(board[r-i][c+i] == piece for i in range(length)):
                    if ((r == rows-1 or c == 0 or board[r+1][c-1] != piece) and
                        (r-length < 0 or c+length == cols or board[r-length][c+length] != piece)):
                        threats += 1

        return threats 

#start env
#make training loop
#------------------------ MAIN ------------------------ #
if __name__ == "__main__":
    env = Connect4Env() 

    #setup agent
    if PLAYER:
        #make player force to not train while playing
        #TRAINMODE = False
        RENDER = True
        RENDER_EACH_MOVE = True
        agent = Connect4Agent_trainer(weights_path=MODEL_FINAL_NAME)
        print("HUMAN VS AI MODE")
    elif not TRAINMODE and not PLAYER:
        RENDER = True
        RENDER_EACH_MOVE = True
        agent = Connect4Agent_trainer(weights_path=MODEL_FINAL_NAME)
        print("random VS AI MODE")
    else:
        agent = Connect4Agent_trainer(weights_path=MODEL_TRAIN_NAME)

    #for smart move player (self play)
    agent_player = Connect4Agent_trainer(weights_path=MODEL_TRAIN_NAME)

    #__________________ START RENDER ____________________
    if RENDER:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        draw_board(env.board)
        pygame.display.update()
        clock = pygame.time.Clock() #for framerate
    else:
        print("Running with no display")


    #some things to store over game time 
    #   episode -> number of games
    #   point results-> score of each player being stored
    #   states -> all states this batch of games that model sees
    #   actions ->  actions the model took to lead to rewards
    #   discounted_rewards -> rewards of batched rewards discounted
    #   loss -> the amount the model learned (to put it simply)
    episodes = 0
    point_results = [] # store points for finding win percent 1 = agent win, 0 = loss/draw
    states, actions, discounted_rewards = [], [], []
    loss = 0
    last_loss = 0 
    
    try:                
        while True:
            episodes += 1
            state = env.reset()
            done = False

            #same as above but for each episode
            episode_states = []
            episode_actions = []
            episode_rewards = []


            # ---------- GAME LOOP  ----------
            while not done:
                if RENDER:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            #leaving = myfont.render("LEAVING",True, (255, 255, 255))
                            #screen.blit(text_surface, (50, 50))
                            #pygame.display.update()
                            pygame.quit()
                            sys.exit()

                # get valid moves
                valid_mask = [env.is_valid_move(c) for c in range(COLUMNS)] #for agent
                valid_indices = [c for c in range(len(valid_mask)) if valid_mask[c]] #for players
                
                if env.current_player == MODEL_PLAYER_ID:
                    action = agent.get_action(state, valid_mask)
                else:#cpu or player
                    if PLAYER:
                        action = None
                        draw_board(env.board)
                        pygame.display.update()

                        #wait for player
                        while action is None:
                            
                            for event in pygame.event.get():
                                #leaving
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                #move circle to place
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
                                #place user piece
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    col = getPosition(event.pos)

                                    #print(f"SELECTED COL: {col}")

                                    if col in valid_indices:
                                        action = col
                                    else:
                                        print("ERROR: invalid move, try again")
                     
                            clock.tick(30) #30 frames a second
                    #self play against self - doesnt work well, agent_player trained off of 
                    elif SMART_MOVE:
                        action = agent_player.get_action(state, valid_mask)
                    #select random valid move
                    else:
                        action = random_move(env)

                # Take a step in the environment
                next_state, reward, done,winner = env.step(action)
                
                
                if RENDER_EACH_MOVE:
                    draw_board(env.board)
                    pygame.display.update()
                    pygame.time.wait(50) #wait a little bit

                if RENDER_TEXT:
                    printBoard(env.board)
                
                #store state, action, reward for round
                if TRAINMODE:
                    episode_states.append(state)
                    episode_actions.append(int(action))
                    episode_rewards.append(float(reward))

                state = next_state
                

            # ---------- EPISODE FINISHED ----------
            # ---------- AFTER GAME ENDS: SHOW FINAL BOARD ----------
            if RENDER:
                draw_board(env.board)
                pygame.display.update()
                pygame.time.wait(50)#wait a little bit

            if RENDER_TEXT:
                printBoard(env.board)

            #discount rewards this round
            if TRAINMODE and len(episode_states) > 0:
                # print(len(episode_states))
                # Discount rewards for THIS episode only
                discounted = discount_rewards(episode_rewards)
                
                # Store episode data (as flat lists)
                states.extend(episode_states)      
                actions.extend(episode_actions)   
                discounted_rewards.extend(discounted)  

                #print(states)
                # Check if we have enough for a batch
                if len(states) >= BATCH_SIZE:
                    
                    # Take first BATCH_SIZE samples
                    batch_states = states[:BATCH_SIZE]
                    batch_actions = actions[:BATCH_SIZE]
                    batch_rewards = discounted_rewards[:BATCH_SIZE]
                    
                    # Train on batch
                    loss = agent.training_step(batch_states, batch_actions, batch_rewards)
                    last_loss = loss

                    # Remove used samples (sliding window)
                    states = states[BATCH_SIZE:]
                    actions = actions[BATCH_SIZE:]
                    discounted_rewards = discounted_rewards[BATCH_SIZE:]
                else:
                    loss = last_loss
            

            #store point for model for win rate
            if winner == MODEL_PIECE:
                point_results.append(1)      # agent win
            else:
                point_results.append(0)      # loss or draw
 

            # # AFTER episode ends
            #total_reward = sum(rewards)

            # ---------- LOGGING
            if PLAYER:
                if winner == MODEL_PIECE:
                    print("Winner: MODEL |")
                else:
                    print("Winner: PLAYER |")
                print(f" Score Player={env.score[0]} Agent={env.score[1]}")
        
            # ---------- LOGGING every 10 episodes (asked chat for some print
            if episodes % 10 == 0 and TRAINMODE:
                recent = point_results[-10:] if len(point_results) >= 10 else point_results
                win_rate = np.mean(recent) * 100
                total_wins = sum(point_results)
                # Fixed: use episode_rewards for last episode's average
                avg_reward_last_ep = np.mean(episode_rewards) if episode_rewards else 0
                # Fixed: loss might not be defined yet
                
                print(f"Episode {episodes} | Score CPU={env.score[0]} Agent={env.score[1]} | "
                    f"Win% (last 10): {win_rate:.1f}% | Total wins: {total_wins} | "
                    f"Avg reward last ep: {avg_reward_last_ep:.3f} | Loss: {loss:.4f}")

            # ---------- LOGGING every 100 episodes (asked chat for some print) ----------
            if episodes % 100 == 0:
                # if TRAINMODE:
                #     SMART_MOVE = not SMART_MOVE
                #     agent_player.model.set_weights(agent.model.get_weights())
                
                win_rate_100 = np.mean(point_results[-100:]) * 100
                print("-"*20)
                print("\n")
                agent.save_model(MODEL_TRAIN_NAME)
                print(f"Points: {len(point_results)} | Score CPU={env.score[0]} Agent={env.score[1]} | "
                      f"Win% (last 100): {win_rate_100:.1f}%")
                
                      
                win_rate_total = np.mean(point_results) * 100 if len(point_results) >= 100 else 0
                print(f"TOTAL AI WIN RATE THIS SESSION: {win_rate_total:.1f}%\n")
                print("-"*20)
                if win_rate_100 >= 80:  # Stop when win rate hits 75%
                    agent.save_model(MODEL_FINAL_NAME)
                    print(f"Target win rate reached! Stopping at episode {episodes}")
                    sys.exit()
                
            
            # if episodes % 200 == 0:
            #     if TRAINMODE:
            #         SMART_MOVE = not SMART_MOVE
            #         if SMART_MOVE:
            #             print("PLAYING SELF")
            #         else:
            #             print("PLAYING RANDOM")
            #         agent_player.model.set_weights(agent.model.get_weights())
                
            
             
    except KeyboardInterrupt:
        print("\nSaving final model and exiting...")
        agent.save_model(MODEL_FINAL_NAME)
        if RENDER:
            pygame.quit()
        sys.exit()
    





