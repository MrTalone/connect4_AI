import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout
import os


ROWS = 6
COLUMNS = 7

class Connect4Agent:
    def __init__(self,weights_path = None):
        self.state_size = 3 * ROWS * COLUMNS
        self.num_actions = COLUMNS
        self.prev_state = None

        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        if weights_path and os.path.exists(weights_path):
            try:
                self.model.load_weights(weights_path)
                print(f"Agent: Weights loaded.")
            except Exception as e:
                print(f"Agent: Failed to load weights: {e}")


    def build_model(self):
        return tf.keras.Sequential([
                Input(shape=(self.state_size,)),
                Dense(16,activation='relu'),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                Dense(self.num_actions, activation='softmax')
            ])
    
    

    def get_action(self,state,valid_moves):
        state = np.array(state, dtype=np.float32)

        # if self.prev_state is None:
        #     x = np.zeros_like(state)
        # else:
        #     x = state - self.prev_state

        # self.prev_state = state

        probs = self.model(state[None, :], training=False).numpy()[0]

         # Mask invalid moves
        mask = np.array(valid_moves, dtype=np.float32)
        probs = probs * mask

        valid_indices = np.where(mask == 1)[0]

        if len(valid_indices) == 0:
            # fallback (should never happen)
            action = np.random.randint(self.num_actions)
            return action

        valid_probs = probs[valid_indices]

        # renormalize
        if np.sum(valid_probs) == 0:
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
        else:
            valid_probs = valid_probs / np.sum(valid_probs)

        action = np.random.choice(valid_indices, p=valid_probs)
        return action

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
            loss = -tf.reduce_mean(log_probs * rewards)
            #policy_loss = -tf.reduce_mean(log_probs * rewards)
            
            #  # --- Entropy bonus (encourage exploration) ---
            # entropy = -tf.reduce_mean(
            #     tf.reduce_sum(probs * tf.math.log(probs+ 1e-8), axis=1)
            # )

            # # Final loss
            # loss = policy_loss - .01 * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()

    def save_model(self, path = "DEF.weights.h5"):
        self.model.save_weights(path)

    def reset(self):
        self.prev_state = None
        

