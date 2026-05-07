#used in connect4_ai.py
#followed cs-421 pong nn
#creates a model to be ran on connect 4 using CNN, DENSE, and RL

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten,BatchNormalization
import os


ROWS = 6
COLUMNS = 7

class Connect4Agent:
    def __init__(self,weights_path = None):
        self.state_size = 3 * ROWS * COLUMNS
        self.num_actions = COLUMNS
        self.prev_state = None

        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=4e-4)
        
        #LOAD PREV WEIGHTS
        print(20*"-")
        if weights_path and os.path.exists(weights_path):
            try:
                self.model.load_weights(weights_path)
                print(f"Agent: Weights loaded.")
            except Exception as e:
                print(f"Agent: Failed to load weights: {e}")
        else:
            print("Agent: no existing weights, creating new model")

    #BUILD CNN MODEL FOR RL
    def build_model(self):
        return tf.keras.Sequential([
            Input(shape=(3, ROWS, COLUMNS)),
            #Reshape((4,ROWS,COLUMNS)),

            Conv2D(32, 5, activation='relu',padding='same'),
            BatchNormalization(),  
            Conv2D(64, 3, activation='relu',padding='same'),
            BatchNormalization(),
            Conv2D(128,3,activation='relu',padding='same'),
            BatchNormalization(),

            Flatten(),
            Dense(256,activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_actions,activation='softmax')
        ])
        # return tf.keras.Sequential([
        #         Input(shape=(self.state_size,)),
        #         Dense(16,activation='relu'),
        #         Dense(64, activation='relu'),
        #         Dense(128, activation='relu'),
        #         Dense(128, activation='relu'),
        #         Dropout(0.3),

        #         Dense(64, activation='relu'),
        #         Dense(self.num_actions, activation='softmax')
        #     ])
    
    
    #USE PROBABILITY TO SUGGEST STRONGEST COL ACTION
    def get_action(self,state,valid_moves,deterministic = False):
        state_input = np.expand_dims(state, axis=0)  

        #state difference      
        # if self.prev_state is None:
        #     x = np.zeros_like(state)
        # else:
        #     x = state - self.prev_state

        # self.prev_state = state


        probs = self.model(state_input, training=False).numpy()[0]

        # # Mask invalid moves
        mask = np.array(valid_moves, dtype=np.float32)
        probs = probs * mask

        # print(probs)

        valid_indices = np.where(mask == 1)[0]

        #no abaible moves available
        if len(valid_indices) == 0:
            # fallback (should never happen)
            action = np.random.randint(self.num_actions)
            return action

        valid_probs = probs[valid_indices]

        valid_probs = np.round(valid_probs, 4) 
        # renormalize
        prob_sum = np.sum(valid_probs)
        if prob_sum == 0:
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
            #action = np.random.choice(valid_indices, p=valid_probs)
        else:
            valid_probs = valid_probs / prob_sum
            #action = valid_indices[np.argmax(valid_probs)]
               
        #print(valid_probs)
        

        action = np.random.choice(valid_indices, p=valid_probs)
        


        return action

    #training the model
    #use states batch from connect4_ai
    #use actions batch from connect4_ai
    #use discount rewards of each episode from connect4_ai
    def training_step(self, states, actions, rewards):
        if len(states) == 0:
            return None

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)

        actions_onehot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs = self.model(states, training=True)
            selected_probs = tf.reduce_sum(probs * actions_onehot, axis=1)
            log_probs = tf.math.log(selected_probs + 1e-8)
            #loss = -tf.reduce_mean(log_probs * rewards)
            policy_loss = -tf.reduce_mean(log_probs * rewards)
            
             # --- Entropy bonus (encourage exploration) ---
            entropy = -tf.reduce_mean(
                tf.reduce_sum(probs * tf.math.log(probs+ 1e-8), axis=1)
            )

            # Final loss
            loss = policy_loss - 0.5 * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        #help from explosion of loss
        #gradient clipping -> prevnt unstable learning and large gradients-> now to max 1
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()

    def save_model(self, path = "DEF.weights.h5"):
        self.model.save_weights(path)
        print(f"Agent: Saved weights to {path}")

    def reset(self):
        self.prev_state = None
        

