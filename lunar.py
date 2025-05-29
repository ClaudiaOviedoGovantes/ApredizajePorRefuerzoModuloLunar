import gymnasium as gym

class LunarLanderEnv:
    def __init__(self, render_mode="human", continuous=False,
        gravity=-10.0, enable_wind=False, 
        wind_power=15.0, turbulence_power=1.5):
        """
        Initialize the Lunar Lander environment with specified parameters.<br><br>
        Parameters:<br>
        render_mode (str): Mode for rendering the environment. Options are 'human', 'rgb_array', or None.<br>
        continuous (bool): If True, use continuous action space. If False, use discrete action space.<br>
        gravity (float): Gravity value for the environment.<br>
        enable_wind (bool): If True, enable wind in the environment.<br>
        wind_power (float): Wind power value.<br>
        turbulence_power (float): Turbulence power value.<br>
        """
            
        self.env = gym.make(
            'LunarLander-v3',
            render_mode=render_mode,
            continuous=continuous,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power
        )
        
        self.reset()
        
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.state, _ = self.env.reset()
        return self.state
    
    # Function to take a single step in the environment
    def take_action(self, action, verbose = False):
        """
        Updates the internal state of the environment based on the action taken.<br><br>
        Parameters:<br>
        action (int or float): The action to be taken in the environment.<br>
        verbose (bool): If True, print the action taken and the resulting state.<br><br>
        Returns:<br>
        state (tuple): The new state of the environment after taking the action.<br>
        reward (float): The reward received after taking the action.<br>
        done (bool): Indicates if the episode has ended.<br><br>
        """
        self.state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if(verbose):
            print(f"Step taken: {action}, New state: {self.state}, Reward: {reward}, Done: {done}")
        
        return self.state, reward, done

    # Function to close the environment
    def close(self):
        self.env.close()
        print("Environment closed.")