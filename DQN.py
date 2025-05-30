import numpy as np
import random
import time

import tensorflow as tf

from collections import deque

from lunar import LunarLanderEnv

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

class DQN(keras.Model):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.dense1 = keras.layers.Dense(hidden_size, activation=keras.activations.leaky_relu)
        self.dense2 = keras.layers.Dense(hidden_size, activation=keras.activations.leaky_relu)
        self.outputLayer = keras.layers.Dense(action_size, activation=keras.activations.leaky_relu)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.outputLayer(x)
    
    #puede requerir mas funciones segun la libreria escogida.
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.001, batch_size=64, 
                memory_size=10000, episodes=1500, 
                target_network_update_freq=10,
                replays_per_episode=1000):
        """
        Initialize the DQN agent with the given parameters.
        
        Parameters:
        lunar (LunarLanderEnv): The Lunar Lander environment instance.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of the batch for experience replay.
        memory_size (int): Number of experiences stored on the replay memory.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        """
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode
        
        # Initialize replay memory
        # a deque is a double sided queue that allows us to append and pop elements from both ends
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize the environment
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        HIDDEN_SIZE = 64

        self.q_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=HIDDEN_SIZE #elegir un tamaño de capa oculta
        )
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=HIDDEN_SIZE #elegir un tamaño de capa oculta
        )
        
        # Set weights of target network to be the same as those of the q network
        self.target_network.set_weights(self.q_network.get_weights())
      
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)# depende del framework que uses (tf o pytorch)
        
        print(f"QNetwork:\n {self.q_network}")
          
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        state = self.lunar.state

        if (random.uniform(0,1) <= self.epsilon):
            # With probability epsilon, choose a random action
            action = self.lunar.env.action_space.sample()

        else:
            # With probability 1 - epsilon
            # Use q-network to evaluate q(s,a) for all actions
            state_tensor = np.array([state], dtype=np.float32)
            result = self.q_network(state_tensor)[0]

            # Choose the action with the highest q(s,a)
            action = np.argmax(result)

    
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Calculate q(s,a) according to the q-network
            # We use one-hot and reduce-sum to keep only the q for the selected action
            #   [ A, B, C, D ] * [ 0, 0, 1, 0 ] -> [ 0, 0, C, 0 ]
            #   reduce_sum([0, 0, C, 0]) -> 0 + 0 + C + 0 = C
            q_values = tf.reduce_sum(self.q_network(states) * tf.one_hot(actions, 4), axis=1)
            
            # Get maximum q-value according to target network for the next state
            # And use it to approximate the next state
            next_q_values = tf.reduce_max(self.target_network(next_states), axis=1)
            
            # Calculate the expected q-values, only adding the next value if done exists
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            loss = tf.keras.losses.MeanSquaredError()(expected_q_values, q_values)
        
        # Train using the gradients
        gradients = tape.gradient(loss, self.q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_weights))

        return loss
        
    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
        pass
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        pass
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        pass
        
    def train(self):
        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """
        
        pass