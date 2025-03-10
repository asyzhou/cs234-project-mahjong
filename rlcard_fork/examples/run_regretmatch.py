import numpy as np
import torch
from tqdm import tqdm

class RegretMatch:
    def __init__(self, ppo_model, action_space_size, learning_rate=0.01, regret_discount=0.95):
        """
        Initialize the Regret Matching agent that uses a PPO model for base policy
        
        Args:
            ppo_model: The trained PPO model (assumed to have actor and critic networks)
            action_space_size: Number of possible actions in the mahjong environment
            learning_rate: Rate at which regrets are updated
            regret_discount: Discount factor for historical regrets
        """
        self.ppo_model = ppo_model
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.regret_discount = regret_discount
        
        # Initialize cumulative regrets
        self.cumulative_regrets = {}
        
        # Set PPO model to evaluation mode to prevent updates
        self.ppo_model.actor.eval()
        self.ppo_model.critic.eval()
    
    def get_state_key(self, state):
        """
        Convert state to a hashable key for the regret table.
        Customize this based on your state representation.
        """
        '''TODO: fix this once ppo is goods'''
        if isinstance(state, np.ndarray):
            return hash(state.tobytes())
        elif isinstance(state, torch.Tensor):
            return hash(state.cpu().numpy().tobytes())
        return hash(str(state))
    
    def initialize_regrets(self, state):
        """
        Initialize regrets for a new state
        """
        state_key = self.get_state_key(state)
        if state_key not in self.cumulative_regrets:
            self.cumulative_regrets[state_key] = np.zeros(self.action_space_size)
    
    def get_regret_matching_policy(self, state):
        """
        Compute a policy based on positive regrets using regret matching
        """
        state_key = self.get_state_key(state)
        self.initialize_regrets(state)
        
        # Get positive regrets
        positive_regrets = np.maximum(self.cumulative_regrets[state_key], 0)
        regret_sum = np.sum(positive_regrets)
        
        # If sum of positive regrets is zero, use uniform strategy
        if regret_sum <= 0:
            return np.ones(self.action_space_size) / self.action_space_size
        
        # Return regret-matching strategy
        return positive_regrets / regret_sum
    
    def select_action(self, state):
        """
        Select an action based on the regret matching policy
        """
        policy = self.get_regret_matching_policy(state)
        action = np.random.choice(self.action_space_size, p=policy)
        return action
    
    def prepare_state(self, state):
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        return state_tensor

    # def compute_counterfactual_values(self, env, state, action_taken):
    #     """
    #     Compute counterfactual values for all actions using PPO's value function
    #     """
    #     with torch.no_grad():
    #         # Get action probabilities from PPO's actor
    #         action_probs = self.ppo_model.actor(state_tensor)
            
    #         # Compute advantage for each action based on PPO's actor probabilities
    #         # This is a heuristic as we don't have true counterfactual values
    #         counterfactual_values = np.zeros(self.action_space_size)
            
    #         for action in range(self.action_space_size):
    #             if action == action_taken:
    #                 continue  # Skip the taken action
                
    #             next_state, reward, done, info = env.step(action)
    #             # Prepare state for PPO model if needed
    #             state_tensor = self.prepare_state(next_state)
    #             # Get value estimate for the state
    #             state_value = self.ppo_model.critic(state_tensor).item()

    #             action_prob = action_probs[0, action].item()
    #             # Scale the state value by the action probability
    #             counterfactual_values[action] = state_value * (1.0 + action_prob)
                
    #         return counterfactual_values
    
    def update_regrets(self, state, action_taken, reward, next_state, done):
        """
        Update regrets based on the observed reward and counterfactual values
        """
        state_key = self.get_state_key(state)
        self.initialize_regrets(state)
        
        # Compute counterfactual values for all actions
        counterfactual_values = self.compute_counterfactual_values(state, action_taken)
        
        # Value of the chosen action is the observed reward plus discounted next state value
        # For terminal states, next state value is 0
        if done:
            actual_value = reward
        else:
            with torch.no_grad():
                if isinstance(next_state, np.ndarray):
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                elif isinstance(next_state, torch.Tensor):
                    next_state_tensor = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
                
                next_state_value = self.ppo_model.critic(next_state_tensor).item()
                actual_value = reward + next_state_value 
        
        # Update regrets for all actions
        for action in range(self.action_space_size):
            if action == action_taken:
                continue  # Skip the taken action
            
            # Regret is the difference between counterfactual value and actual value
            regret = counterfactual_values[action] - actual_value
            self.cumulative_regrets[state_key][action] += self.learning_rate * regret
        
        # Apply regret discount
        self.cumulative_regrets[state_key] *= self.regret_discount
    
    def train(self, env, num_episodes=1000, max_steps_per_episode=100, 
              eval_interval=100, eval_episodes=10, save_path=None, verbose=True):
        """
        Train the regret matching algorithm for a specified number of episodes
        
        Args:
            env: The environment to train in
            num_episodes: Total number of training episodes
            max_steps_per_episode: Maximum steps per episode
            eval_interval: Interval to evaluate current policy
            eval_episodes: Number of episodes for evaluation
            save_path: Path to save the regret table periodically
            verbose: Whether to display progress information
        
        Returns:
            dict: Training statistics
        """
        stats = {
            'episode_rewards': [],
            'eval_rewards': [],
            'regret_table_sizes': [],
            'unique_states_visited': set()
        }
        
        progress_bar = tqdm(range(num_episodes)) if verbose else range(num_episodes)
        for episode in progress_bar:
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            # Track unique states
            stats['unique_states_visited'].add(self.get_state_key(state))
            
            while not done and steps < max_steps_per_episode:
                # Select action using regret matching
                action = self.select_action(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Update regrets
                self.update_regrets(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Track unique states
                stats['unique_states_visited'].add(self.get_state_key(state))
            
            # Record episode statistics
            stats['episode_rewards'].append(episode_reward)
            stats['regret_table_sizes'].append(len(self.cumulative_regrets))
            
            # Update progress bar if verbose
            if verbose:
                avg_reward = np.mean(stats['episode_rewards'][-100:]) if len(stats['episode_rewards']) >= 100 else np.mean(stats['episode_rewards'])
                progress_bar.set_description(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | States: {len(self.cumulative_regrets)}")
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(env, num_episodes=eval_episodes)
                stats['eval_rewards'].append((episode + 1, eval_reward))
                if verbose:
                    print(f"\nEvaluation at episode {episode+1}: Average reward: {eval_reward:.2f}\n")
                    
                # Save regret table if path is provided
                if save_path:
                    self.save(f"{save_path}_episode_{episode+1}.npy")
        
        # Convert unique states to count at the end
        stats['unique_states_visited'] = len(stats['unique_states_visited'])
        
        # Final save
        if save_path:
            self.save(f"{save_path}_final.npy")
            
        return stats
    
    def evaluate(self, env, num_episodes=10, render=False):
        """
        Evaluate the current policy
        
        Args:
            env: Environment to evaluate in
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            float: Average reward across evaluation episodes
        """
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action using current policy (no exploration)
                policy = self.get_regret_matching_policy(state)
                action = np.argmax(policy)  # Greedy action selection for evaluation
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                if render:
                    env.render()
                
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)

        return np.mean(total_rewards)
    
    def save(self, path):
        """Save the regret table to a file"""
        np.save(path, self.cumulative_regrets)
    
    def load(self, path):
        """Load the regret table from a file"""
        self.cumulative_regrets = np.load(path, allow_pickle=True).item()


# Example usage
def example_usage(ppo_model, env):
    action_space_size = env.action_space.n
    rm_agent = RegretMatch(ppo_model, action_space_size)
    
    # Train the agent
    stats = rm_agent.train(
        env=env,
        num_episodes=500,
        eval_interval=50,
        save_path="regret_matching_checkpoints"
    )
    
    # Evaluate final policy
    final_reward = rm_agent.evaluate(env, num_episodes=20, render=True)
    print(f"Final evaluation average reward: {final_reward:.2f}")
    
    # Save the final model
    rm_agent.save("regret_matching_final.npy")