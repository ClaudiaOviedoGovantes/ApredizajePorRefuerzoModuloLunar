from lunar import LunarLanderEnv

# import torch
# import tensorflow as tf

# import numpy as np


class PolicyNetwork(torch.nn.Module/tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size):
        pass
    
    #puede requerir mas funciones segun la libreria escogida.
    

class REINFORCEAgent:
    def __init__(self, lunar: LunarLanderEnv, learning_rate=0.001,
        gamma=0.99, episodes=5000):
        
        self.episodes = episodes
        self.lunar = lunar
        self.discount = gamma
        
        self.action_size = lunar.env.action_space.n
        self.policy = PolicyNetwork(lunar.env.observation_space.shape[0], 
                                       self.action_size, 
                                       hidden_size=0) #elegir un tamaño de capa oculta
        
        self.optimizer = # depende del framework que uses (tf o pytorch)

    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it is taken from the policy network and randomly sampled from the action space.
        """
        pass
        
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        
        return next_state, reward, done, action
    
    def compute_discounted_rewards(self, rewards):
        """
        Compute the discounted rewards for a given trajectory.
        Parameters:
        rewards (list): List of rewards for each time step in the trajectory.
        Returns:
        np.ndarray: Array of discounted rewards for each time step in the trajectory.
        """ 
       
        # turn rewards into return we backpropagate the rewards and multiply them by the discount factor
        # to get the discounted return for each state-action pair
        # this is done in reverse order, so we start from the end of the trajectory
        
            
        return discounted_rewards
    
    def replay_experience(self, states, actions, rewards):
        """
        Replay the experience of the agent and update the policy network.
        Parameters:
        states (list): List of states in the trajectory.
        actions (list): List of actions taken in the trajectory.
        rewards (list): List of rewards received in the trajectory.
        
        Returns:
        float: The loss value from the policy network update.
        """
        
        disc_rew = self.compute_discounted_rewards(rewards)
            
        pass
        
        return loss
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        pass
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        pass

    def train(self):
        """
        Train the agent using the REINFORCE algorithm.
        The agent interacts with the environment, collects experiences, and updates the policy network.
        The training continues until the agent reaches a specified number of episodes or achieves a certain performance.
        The training process MUST be logged and the model is saved periodically.
        """
        
        pass