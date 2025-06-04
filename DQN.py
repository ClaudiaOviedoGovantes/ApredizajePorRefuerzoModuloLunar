import pytorch_amd_hack_setup
import numpy as np
import torch
from collections import deque
import random

import os

from lunar import LunarLanderEnv

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()

        self.input_layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer1_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer2_output = torch.nn.Linear(hidden_size, action_size)
    
    def call(self, input):
        return self.forward(input)

    def forward(self, input):
        x = torch.nn.functional.leaky_relu(self.input_layer1(input))
        x = torch.nn.functional.leaky_relu(self.layer1_layer2(x))
        return self.layer2_output(x)
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000, state_size=8):
        self.len = 0
        self.pos = 0
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size,), dtype=np.int64)
        self.reward_buffer = np.zeros((buffer_size,), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.state_buffer[i] = state
        self.action_buffer[i] = action
        self.reward_buffer[i] = reward
        self.next_state_buffer[i] = next_state
        self.done_buffer[i] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)
        
    def sample(self, batch_size):
        if batch_size == 0 or batch_size > self.len:
            return None

        choices = np.random.choice(self.len, batch_size, replace=False)

        states = self.state_buffer[choices]
        actions = self.action_buffer[choices]
        rewards = self.reward_buffer[choices]
        next_states = self.next_state_buffer[choices]
        dones = self.done_buffer[choices]
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.len

class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.001, batch_size=64, 
                memory_size=10000, episodes=1500, 
                target_network_update_freq=10,
                replays_per_episode=1000, hidden_size=64, memory=None):
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
        if memory == None:
            self.memory = ReplayBuffer(memory_size)
        else:
            self.memory = memory
        
        # Initialize the environment
        self.lunar = lunar
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        self.observation_dims = lunar.env.observation_space.shape[0]
        self.action_size = lunar.env.action_space.n
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(
            state_size=self.observation_dims,
            action_size=self.action_size,
            hidden_size=self.hidden_size #elegir un tamaño de capa oculta
        ).to(self.device)
        
        self.target_network = DQN(
            state_size=self.observation_dims,
            action_size=self.action_size,
            hidden_size=self.hidden_size #elegir un tamaño de capa oculta
        ).to(self.device).eval()
        
        # Set weights of target network to be the same as those of the q network
        self.update_target_network()
      
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)# depende del framework que uses (tf o pytorch)
        
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
            self.q_network.eval()

            state_tensor = torch.tensor(state).to(self.device).float()

            with torch.no_grad():
                result = self.q_network(state_tensor).cpu().numpy()

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
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        self.q_network.train()

        # Calculate q(s,a) according to the q-network
        # We use one-hot and reduce-sum to keep only the q for the selected action
        #   [ A, B, C, D ] * [ 0, 0, 1, 0 ] -> [ 0, 0, C, 0 ]
        #   reduce_sum([0, 0, C, 0]) -> 0 + 0 + C + 0 = C
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get maximum q-value according to target network for the next state
        # And use it to approximate the next state
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # Calculate the expected q-values, only adding the next value if done exists
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = torch.nn.MSELoss()(q_values, expected_q_values)

        # Train using the gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
        
    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        torch.save(self.target_network.state_dict(), path)
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        state_dict = torch.load(path, weights_only=True)

        if (state_dict == None):
            raise ValueError(f"Tried to load model \"{path}\" resulted in None!")
        
        self.q_network.load_state_dict(state_dict)

        self.update_target_network()
    
    def train(self, save_path="modelo_DQN.h5", score_window_size=50, backup_interval=50):
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
        
        with open(f"{path}/log.txt", "a+") as log_file:
            def log(message):
                """
                Log a message to the console and to a file.
                """
                timestamp = np.datetime64('now', 's')
                message = f"[{timestamp}] {message}"
                print(message)
                log_file.write(message + "\n")
            
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
            steps = 0

            for episode in range(0, self.episodes):

                self.lunar.reset()

                score = 0

                done = False
                while not done:
                    _next_state, reward, done, _action = self.act()
                    score += reward
                    steps += 1

                    if (steps % self.target_updt_freq == 0):
                        self.update_model(episode=episode)
                        self.update_target_network()
                    
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                epsilons.append(self.epsilon)
                score_buffer.append(score)
                scores.append(score)
                
                if (episode % backup_interval == 0):
                    average_score = np.mean(score_buffer)
                    averaged_scores.append(average_score)

                    percentage = 100*(episode+1)/self.episodes
                    log(f"Episode {episode+1} ({percentage:.2f}%) had score: {average_score:.2f}")
                    backup_path = f"{path}/episode_{episode+1}_({average_score:.2f}).h5"
                    log(f"    Saving model to {backup_path}")
                    self.save_model(backup_path)
                
                
            log("Saving scores, averaged scores and epsilons to files...")

            scores_path = f"{path}/summary_scores.csv"
            averaged_scores_path = f"{path}/summary_averaged_scores.csv"
            epsilons_path = f"{path}/summary_epsilons.csv"

            with open(scores_path, "w") as scores_file:
                scores_file.write("episode,score\n")

                for i, score in enumerate(scores):
                    scores_file.write(f"{i+1},{score}\n")

            with open(averaged_scores_path, "w") as averaged_scores_file:
                averaged_scores_file.write("episode,average_score\n")

                for i, average_score in enumerate(averaged_scores):
                    averaged_scores_file.write(f"{i*backup_interval+1},{average_score}\n")

                if((steps-1)%backup_interval != 0):
                    averaged_scores_file.write(f"{steps-1},{np.mean(score_buffer)}\n")

            with open(epsilons_path, "w") as epsilons_file:
                epsilons_file.write("episode,epsilon\n")

                for i, epsilon in enumerate(epsilons):
                    epsilons_file.write(f"{i+1},{epsilon}\n")
            
            backup_path = f"{path}/final_({average_score:.2f}).h5"
            log(f"Training finished! Saving as \"{backup_path}\"")
            self.save_model(backup_path)

            print(f"Training finished! Saving as \"{save_path}\"")
            self.save_model(save_path)
    
class DQNAgentDoubleDPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size=10000, state_size=8, alpha=0.6):
        super().__init__(buffer_size=buffer_size, state_size=state_size)
        self.alpha = alpha
        self.priority_buffer = np.zeros((buffer_size,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        max_priority = np.max(self.priority_buffer[:self.len]) if self.len > 0 else 1.0
        self.priority_buffer[i] = max_priority  

        super().push(state, action, reward, next_state, done)
        
    def sample(self, batch_size, beta):
        if batch_size == 0 or batch_size > self.len:
            return None

        probabilities = self.priority_buffer[:self.len] / np.sum(self.priority_buffer[:self.len])

        choices = np.random.choice(self.len, batch_size, replace=False, p=probabilities)

        weights = (self.len * probabilities[choices]) ** (-beta)
        weights /= weights.max()

        states = self.state_buffer[choices]
        actions = self.action_buffer[choices]
        rewards = self.reward_buffer[choices]
        next_states = self.next_state_buffer[choices]
        dones = self.done_buffer[choices]
        return choices, states, actions, rewards, next_states, dones, weights
    
    def update_priorities(self, indices, losses):
        clipped = np.maximum(losses, 0.01)
        self.priority_buffer[indices] = losses**self.alpha
    
    def __len__(self):
        return self.len

class DQNAgentDoubleDPrioritizedReplay(DQNAgent):
    def __init__(self, alpha=0.6, initial_beta=0.4, memory_size=10000, **kwargs):
        super().__init__(**kwargs, memory_size=memory_size,memory=DQNAgentDoubleDPrioritizedReplayBuffer(
            buffer_size=memory_size,
            alpha=alpha
        ))

        self.initial_beta = initial_beta

    def update_model(self, episode=0):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
        beta = min(self.initial_beta + (1 - self.initial_beta) * episode / self.episodes, 1)
        
        sample = self.memory.sample(self.batch_size, beta=beta)

        if (sample == None):
            return None

        choices, states, actions, rewards, next_states, dones, weights = sample
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)

        self.q_network.train()

        # Calculate q(s,a) according to the q-network
        # We use one-hot and reduce-sum to keep only the q for the selected action
        #   [ A, B, C, D ] * [ 0, 0, 1, 0 ] -> [ 0, 0, C, 0 ]
        #   reduce_sum([0, 0, C, 0]) -> 0 + 0 + C + 0 = C
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(-1)
        
        # Get maximum q-value according to target network for the next state
        # And use it to approximate the next state
        with torch.no_grad():
            arg_max = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, arg_max.unsqueeze(1)).squeeze(-1)
        
        # Calculate the expected q-values, only adding the next value if done exists
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        errors = q_values - expected_q_values # Calculate error not squared, to use it for the priorities

        new_priorities = errors.abs().detach().cpu().numpy()
        self.memory.update_priorities(choices, new_priorities)

        loss = (weights * errors.pow(2)).mean() # Error squared is MSE
        
        # Train using the gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
