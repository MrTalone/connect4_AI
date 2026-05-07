# Connect 4 Reinforcement Learning AI

A Python-based Connect 4 game that uses Reinforcement Learning (RL) and a Convolutional Neural Network (CNN) to train an AI agent to play Connect 4.

This project supports:

- AI Training
- Human vs AI Gameplay
- AI vs Random Opponent
- Experimental Self-Play

------------------------------------------------------------------

# Features

- Connect 4 environment built from scratch
- CNN-based reinforcement learning agent
- Pygame graphical interface
- Reward-based policy gradient learning
- Automatic model saving/loading
- Multiple gameplay and training modes

------------------------------------------------------------------


# Project Files

| File | Description |

- `connect4_ai.py` | Main game loop, environment, rendering, training |
- `connect4_agent.py` | Neural network agent and RL training logic |

------------------------------------------------------------------

# Requirements

Install the required libraries:
 `pip install numpy pygame tensorflow `

------------------------------------------------------------------

# How to Run

- python connect4_ai.py

------------------------------------------------------------------

#Game Modes

- All setting are controlled near top of connect4_ai.py

------------------------------------------------------------------
# Training mode

- TRAINMODE = True
- PLAYER = False
- SMART_MOVE = False

- Train the AI against random opponent
- save model checkpoints automatically
- stops when targe win rate is reached

output:
-Talon_Connect4.weights.h5        - current training checkpoint
-Talon_Connect4_Final.weights.h5  - final trained model

------------------------------------------------------------------
# Human vs AI Mode

- TRAINMODE = False
- PLAYER = True
- SMART_MOVE = False

- Play against the trained AI
- uses mouse controls
- loads final saved model

CONTROLS:
- move piece - Move Mouse
- Drop Piece - Left Click

NOTE: make sure a Connect4_Final.weights.h5 exist first

------------------------------------------------------------------

# AI vs Random CPU

- TRAINMODE = False
- PLAYER = False
- SMART_MOVE = False

- Ai plays against a random opp
- Useful for testing model

------------------------------------------------------------------

# self play mode (Experimental)

- TRAINMODE = True
- PLAYER = False
- SMART_MOVE = True

- AI plays against another Ai
- Experimental training

------------------------------------------------------------------

# Render Options

- RENDER = True       - Display pygame window
- RENDER_TEXT = True  - Prints board states to the terminal.

------------------------------------------------------------------

# Board settings

- ROWS = 6
- COLUMNS = 7
- CONNECT = 4

------------------------------------------------------------------

# Neural Network Architecture

- Convolutional Neural Networks (CNN)
- Dense Layers
- Batch Normalization
- Dropout

--- Policy Gradient Reinforcement Learning ---

# Input Channels

| 0 - Human/player pieces   |
| 1 - AI Piece              |
| 2 - Current player's turn |

# Rewards

AI Win  -> +5
AI Loss -> -5
Draw -> +0.1
Create Threats -> +
opponent Threats -> -
invalid move -> -1

------------------------------------------------------------------

#Save models

Talon_Connect4.weights.h5        -  Saves every 100 episodes
Talon_Connect4_Final.weights.h5  - Saves when either - Training ends - CTRL+C - Target Win Rate Reached

------------------------------------------------------------------

# NOTES

Training is much faster with rendering disabled
TensorFlow may automatically use GPU acceleration
Self-play mode is experimental
First-time training does not require weight files

------------------------------------------------------------------

# Credits

Created by Talon Finehout

Based on:
  Keith Galli Connect 4 tutorial series
  CS-421 Pong reinforcement learning examples

  Keith Galli Playlist:
    https://www.youtube.com/playlist?list=PLFCB5Dp81iNV_inzM-R9AKkZZlePCZdtV
    
