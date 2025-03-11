import numpy as np
import torch
from tqdm import tqdm
from tensordict import TensorDict
from rlcard.envs.mahjong import MahjongEnv
from rlcard.envs.mahjong_envwrap import MahjongTorchEnv
from run_ppo import PPOActorCritic
import copy


class RegretMatch:
    def __init__(self, ppo_model, action_space_size, learning_rate=0.01, regret_discount=0.95):
        """
        Initialize the Regret Matching obj with pretrained PPO model's value function
        
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
        """
        # ideally state is always a tensordict, hash the observation tensor
        if isinstance(state, TensorDict):  
            if 'observation' in state:
                return hash(state['observation'].cpu().detach().numpy().tobytes())
        # else try to hash the entire state
        return hash(str(state))
    
    def initialize_regrets(self, state):
        """
        Initialize regrets for a new state
        """
        state_key = self.get_state_key(state)
        if state_key not in self.cumulative_regrets:
            self.cumulative_regrets[state_key] = np.zeros(self.action_space_size)
    
    def get_regret_matching_policy(self, state, legal_mask=None):
        """
        Compute a policy based on positive regrets using regret matching
        
        Args:
            state: Current state
            legal_mask: Binary mask of legal actions (1 for legal, 0 for illegal)
        """
        state_key = self.get_state_key(state)
        self.initialize_regrets(state)
        
        # Get positive regrets
        positive_regrets = np.maximum(self.cumulative_regrets[state_key], 0)
        # Apply legal action mask if provided
        if legal_mask is not None:
            if isinstance(legal_mask, torch.Tensor):
                legal_mask = legal_mask.cpu().numpy()
            # Zero out regrets for illegal actions
            positive_regrets = positive_regrets * legal_mask
        regret_sum = np.sum(positive_regrets)
        
        # If sum of positive regrets is zero, use uniform strategy over legal actions
        if regret_sum <= 0:
            if legal_mask is not None:
                # Uniform distribution over legal actions
                legal_count = np.sum(legal_mask)
                if legal_count > 0:
                    policy = np.zeros(self.action_space_size)
                    policy = legal_mask / legal_count
                    return policy
            return np.ones(self.action_space_size) / self.action_space_size
        
        # Return regret-matching strategy
        return positive_regrets / regret_sum
    
    def select_action(self, state):
        """
        Select an action based on the regret matching policy
        
        Args:
            state: Current state (can be TensorDict)
        """
        # Extract legal mask in TensorDict
        legal_mask = state['legal_mask']
        policy = self.get_regret_matching_policy(state, legal_mask)
        
        # Ensure the policy sums to 1 jic
        policy = policy / np.sum(policy)
        action = np.random.choice(self.action_space_size, p=policy)
        return action
    
    def compute_true_counterfactual_values(self, env, state):
        """
        Compute true counterfactual values by taking each action in the environment then restoring the original state.
        
        Args:
            env: Environment to use for simulation
            state: Current state
            
        Returns:
            np.ndarray: Array of values for each action
        """
        # Extract legal mask
        legal_mask = None
        legal_mask = state['legal_mask']
        if isinstance(legal_mask, torch.Tensor):
            legal_mask = legal_mask.cpu().numpy()
        
        counterfactual_values = np.full(self.action_space_size, float('-inf'))
        
        # Try each action
        for action in range(self.action_space_size):
            # Skip illegal actions
            if legal_mask[action] == 0:
                continue
                
            # Take the action
            next_state = env.step(action)
            
            # Extract reward and done from TensorDict
            reward = next_state['reward', 0.0]
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            done = next_state['done']
            if isinstance(done, torch.Tensor):
                done = done.item()

            # Calculate counterfactual value
            if done:
                counterfactual_values[action] = reward
            else:
                with torch.no_grad():
                    next_state_tensor = next_state['observation']
                    # Ensure state tensor has batch dimension
                    if next_state_tensor.dim() == 1:
                        next_state_tensor = next_state_tensor.unsqueeze(0)
                    
                    next_state_value = self.ppo_model.critic(next_state_tensor).item()
                    counterfactual_values[action] = reward + next_state_value
            
            # Step back
            env.step_back()
            
        return counterfactual_values
    
    def update_regrets(self, env, state, action_taken, next_state, counterfactual_values):
        """
        Update regrets based on the observed reward and true counterfactual values
        
        Args:
            env: Environment to use for simulation
            state: Current state
            action_taken: Action that was actually taken
            next_state: Resulting state after taking action_taken
        """
        state_key = self.get_state_key(state)
        self.initialize_regrets(state)

        # Calculate the observed reward for the next state
        reward = next_state['reward', 0.0]
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        done = next_state['done']
        if isinstance(done, torch.Tensor):
            done = done.item()

        # Update actual next state value in counterfactual dict jic
        if done:
            counterfactual_values[action_taken] = reward
        else:
            with torch.no_grad():
                next_state_tensor = next_state['observation']
                # Ensure state tensor has batch dimension
                if next_state_tensor.dim() == 1:
                    next_state_tensor = next_state_tensor.unsqueeze(0)
                    
                next_state_value = self.ppo_model.critic(next_state_tensor).item()
                counterfactual_values[action_taken] = reward + next_state_value
        
        # Update regrets for all actions
        for action in range(self.action_space_size):
            if action == action_taken or counterfactual_values[action] == float('-inf'):
                continue  # Skip the taken action and illegal actions
            
            # Regret is the difference between counterfactual value and actual value
            regret = counterfactual_values[action] - counterfactual_values[action_taken]
            self.cumulative_regrets[state_key][action] += self.learning_rate * regret
        # Apply regret discount
        self.cumulative_regrets[state_key] *= self.regret_discount

    
    def train(self, env, num_episodes=1000, max_steps_per_episode=1000, 
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
            episode_reward = 0
            steps = 0
            
            # Track unique states
            stats['unique_states_visited'].add(self.get_state_key(state))
            
            done = False
            while not done and steps < max_steps_per_episode:
                # Compute true counterfactual values for all actions
                counterfactual_values = self.compute_true_counterfactual_values(env, state)

                # Select action using regret matching
                action = self.select_action(state)
                # Take action in environment
                next_state = env.step(action)
                # Calculate the observed reward for the next state
                reward = next_state['reward', 0.0]
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                done = next_state['done']
                if isinstance(done, torch.Tensor):
                    done = done.item()
                
                # Update regrets using true counterfactual values
                self.update_regrets(env, state, action, next_state, counterfactual_values)
                
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
            episode_reward = 0
            done = False
            
            while not done:
                # Extract legal mask
                legal_mask = state['legal_mask']
                
                # Select action using current policy (greedy for evaluation)
                policy = self.get_regret_matching_policy(state, legal_mask)
                action = np.argmax(policy)  # Greedy action selection for evaluation
                
                # Take action in environment
                next_state = env.step(action)
                
                # Extract reward and done from TensorDict
                reward = next_state['reward']
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                done = next_state['done']
                if isinstance(done, torch.Tensor):
                    done = done.item()
                
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


# Basically a train wrapper
def train_example(ppo_model, env):
    action_space_size = env.action_spec['action'].n
    rm_agent = RegretMatch(ppo_model, action_space_size)
    rm_agent.train(
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

if __name__ == "__main__":
    # load ppo model, initialize environment
    mahjong_config = {
        'allow_step_back': True,
        'num_players': 4,
        'seed': 0, # manually add seed here perhap
        'device': 'cpu'
    }
    mahjong_env = MahjongEnv(mahjong_config)
    base_env = MahjongTorchEnv(mahjong_env, device='cpu')
    obs_shape = base_env.observation_spec["observation"].shape
    num_actions = base_env.action_spec['action'].n
    
    ppo = PPOActorCritic(obs_shape, num_actions)
    ppo.load_state_dict(torch.load("model.pth", map_location="cpu"))
    ppo.to("cpu") 
    
    train_example(ppo, base_env) # do i need to wrap this with self play to evaluate ppo on it???