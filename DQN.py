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
    def __init__(self, state_size, action_size, hidden_size, **kwargs):
        super().__init__(**kwargs)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.dense1 = keras.layers.Dense(hidden_size, activation=keras.activations.leaky_relu)
        self.dense2 = keras.layers.Dense(hidden_size, activation=keras.activations.leaky_relu)
        self.outputLayer = keras.layers.Dense(action_size)

        self(tf.ones((1,state_size)))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.outputLayer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        if (batch_size <= len(self.buffer)):
            states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            return states, actions, rewards, next_states, dones

        return None
    
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
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        self.observation_dims = lunar.env.observation_space.shape[0]
        self.action_size = lunar.env.action_space.n
        HIDDEN_SIZE = 64

        self.q_network = DQN(
            state_size=self.observation_dims,
            action_size=self.action_size,
            hidden_size=HIDDEN_SIZE #elegir un tamaño de capa oculta
        )
        
        self.target_network = DQN(
            state_size=self.observation_dims,
            action_size=self.action_size,
            hidden_size=HIDDEN_SIZE #elegir un tamaño de capa oculta
        )
        
        # Set weights of target network to be the same as those of the q network
        self.update_target_network()
      
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)# depende del framework que uses (tf o pytorch)
        
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

        self.memory.push(state, action, reward, next_state, done)

        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
        
        sample = self.memory.sample(self.batch_size)

        if (sample == None):
            return None

        states, actions, rewards, next_states, dones = sample

        with tf.GradientTape() as tape:
            # Calculate q(s,a) according to the q-network
            # We use one-hot and reduce-sum to keep only the q for the selected action
            #   [ A, B, C, D ] * [ 0, 0, 1, 0 ] -> [ 0, 0, C, 0 ]
            #   reduce_sum([0, 0, C, 0]) -> 0 + 0 + C + 0 = C
            q_values = tf.reduce_sum(self.q_network(states) * tf.one_hot(actions, self.action_size), axis=1)
            
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
        self.target_network.set_weights(self.q_network.get_weights())
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        keras.saving.save_model(self.target_network, path, overwrite=True)
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        buffer = keras.saving.load_model(path, {"DQN": DQN})

        if (buffer == None):
            raise ValueError(f"Tried to load model \"{path}\" resulted in None!")
        self.q_network = buffer

        self.update_target_network()
        
    def train(self, save_path="modelo_DQN.h5", score_window_size=100, backup_interval=50):
        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """

        """
        Relevant parameters:
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        """

        # For every episode:
        # Until the simulation is done:
        #   1. Act
        #   2. Use a batch to update the model
        #   3. Update the target network every freq steps
        #   4. Decay epsilon

        training_code = random.randint(0, 1000000)

        print(f"Starting training (code {training_code})...")
        # Create folder training/training_code if it doesn't exist
        path = f"training/training_{training_code}"
        os.makedirs(path)
        
        with open(f"{path}/log.txt", "a+") as f:
            def log(message):
                """
                Log a message to the console and to a file.
                """
                print(message)
                f.write(message + "\n")
            
            log(
                f"Training DQN agent with parameters:\n"
                f"  - gamma: {self.gamma}\n"
                f"  - epsilon: {self.epsilon}\n"
                f"  - epsilon_decay: {self.epsilon_decay}\n"
                f"  - epsilon_min: {self.epsilon_min}\n"
                f"  - learning_rate: {self.learning_rate}\n"
                f"  - batch_size: {self.batch_size}\n"
                f"  - episodes: {self.episodes}\n"
                f"  - target_network_update_freq: {self.target_updt_freq}\n"
                f"  - replays_per_episode: {self.replays_per_episode}\n"
            )

            score_buffer = deque(maxlen=score_window_size)
            scores = []
            averaged_scores = []
            epsilons = []

            for episode in range(0, self.episodes):
                if (episode % 10 == 0):
                    print(f"Episode #{episode+1} ({(episode+1)/self.episodes*100:.2f}%)")

                epsilons.append(self.epsilon)

                self.lunar.reset()

                score = 0

                done = False
                while not done:
                    _next_state, reward, done, _action = self.act()
                    self.update_model()

                    if (len(self.memory) >= self.batch_size):
                        pass

                    score += reward
                
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                score_buffer.append(score)
                scores.append(score)

                if (episode % self.target_updt_freq == 0):
                    self.update_target_network()
                if (episode % backup_interval == 0):
                    average_score = np.mean(score_buffer)
                    averaged_scores.append(average_score)

                    log(f"Episode {episode+1} had score: {average_score:.2f}")
                    backup_path = f"{path}/episode_{episode+1}_({average_score:.2f}).h5"
                    log(f"Saving model to {backup_path}")
                    self.save_model(backup_path)
            
            log("Saving scores, averaged scores and epsilons to files...")

            scores_path = f"{path}/summary_scores.csv"
            averaged_scores_path = f"{path}/summary_averaged_scores.csv"
            epsilons_path = f"{path}/summary_epsilons.csv"
            with open(scores_path, "w") as f:
                f.write("episode,score\n")
                for i, score in enumerate(scores):
                    f.write(f"{i+1},{score}\n")
            with open(averaged_scores_path, "w") as f:
                f.write("episode,average_score\n")
                for i, average_score in enumerate(averaged_scores):
                    f.write(f"{i*backup_interval+1},{average_score}\n")
                if((self.episodes-1)%backup_interval != 0):
                    f.write(f"{self.episodes},{np.mean(score_buffer)}\n")
            with open(epsilons_path, "w") as f:
                f.write("episode,epsilon\n")
                for i, epsilon in enumerate(epsilons):
                    f.write(f"{i+1},{epsilon}\n")
            
            print(f"Training finished! Saving as \"{save_path}\"")
            self.save_model(save_path)
