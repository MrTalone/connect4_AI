#TALON FINEHOUT

#this connect 4 game follows some setup from keith Galli on youtube
#How to Program Connect 4 in Python!(1-4)
#https://www.youtube.com/playlist?list=PLFCB5Dp81iNV_inzM-R9AKkZZlePCZdtV

import numpy as np
import pygame
import sys
import math

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
    return

def printBoard(board):
    print(np.flip(board,0))

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


def gameOverBoard():
    global userTurn, board, gameOver
    userTurn=0
    board[:,:] = ' '
    gameOver=False
    printBoard(board)

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

startGraphics = input("DO YOU WANT GRAPHIC (y/n): ").strip().lower()
if(startGraphics == "y"):
        pygame.init()

        SQUARE_SIZE = 100
        RADIUS = int(SQUARE_SIZE/2-6)
        width = COLUMNS*SQUARE_SIZE
        height = (ROWS+1)*SQUARE_SIZE
        screenSize = (width,height)
        screen = pygame.display.set_mode(screenSize)

        draw_board(board)
        pygame.display.update()

def getPosition(pos):
    posx = pos[0]
    return int(math.floor(posx/SQUARE_SIZE))

def playWithGraphics():
    global userTurn, board, gameOver
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            #print(event.pos)
            pygame.draw.rect(
                screen,
                (0,0,0),
                (0,0,width,SQUARE_SIZE)
            )

            if userTurn == 0:
                pygame.draw.circle(
                    screen,
                    (255,0,0),
                    (
                        (event.pos[0],
                        int(SQUARE_SIZE/2))
                    ),
                    int(RADIUS)
                )
            else:
                pygame.draw.circle(
                    screen,
                    (0,200,100),
                    (
                        (event.pos[0],
                        int(SQUARE_SIZE/2))
                    ),
                    int(RADIUS)
                )
            #pygame.display.update()


        if event.type == pygame.MOUSEBUTTONDOWN:
            print(event.pos)
            #player input
            if userTurn == 0:
                col = getPosition(event.pos)
                print(col)

                if isValidInput(board,col):
                    row = getOpenRow(board,col)
                    dropPiece(board,row,col,PLAYER_PIECE)

                    if checkWin(board,PLAYER_PIECE):
                        print("Player 1 win")
                        gameOver = True
    
            #for start make user input
            else:
                col = getPosition(event.pos)
                print(col)

                if isValidInput(board,col):
                    row = getOpenRow(board,col)
                    dropPiece(board,row,col,MODEL_PIECE)

                    if checkWin(board,MODEL_PIECE):
                        print("Player 2 win")
                        gameOver = True
            
        
            printBoard(board)

            userTurn += 1
            userTurn = userTurn % 2
        draw_board(board)
        pygame.display.update()


while not gameOver:

    if(startGraphics == "y"):
        playWithGraphics()    
    #regualar play with text
    else:
        #player input
        if userTurn == 0:
            col = int(input(f"Make selection (0-{COLUMNS-1}): "))
            print(col)

            if isValidInput(board,col):
                row = getOpenRow(board,col)
                dropPiece(board,row,col,PLAYER_PIECE)

                if checkWin(board,PLAYER_PIECE):
                    print("Player 1 win")
                    gameOver = True
        
        #for start make user input
        else:
            col = int(input(f"Make selection (0-{COLUMNS-1}): "))
            print(col)

            if isValidInput(board,col):
                row = getOpenRow(board,col)
                dropPiece(board,row,col,MODEL_PIECE)

                if checkWin(board,MODEL_PIECE):
                    print("Player 2 win")
                    gameOver = True
        

        printBoard(board)

        userTurn += 1
        userTurn = userTurn % 2

    if gameOver:
        print("Game resetting...")
        pygame.time.wait(2000) 
        gameOverBoard()


        






            
