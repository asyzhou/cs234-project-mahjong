import numpy as np
import torch
from tqdm import tqdm
from tensordict import TensorDict
from rlcard.envs.mahjong import MahjongEnv
from rlcard.envs.mahjong_envwrap import MahjongTorchEnv
from run_ppo import PPOActorCritic
import copy
import argparse
import wandb
import os
import time
from datetime import datetime


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
        self.count = 0
        # Set PPO model to evaluation mode to prevent updates
        self.ppo_model.policy.eval() # actor
        self.ppo_model.value.eval() # critic
    
    def get_state_key(self, state, reduced=True):
        """
        Convert state to a hashable key for the regret table.
        """
        # ideally state is always a tensordict, hash the observation tensor
        # if isinstance(state, TensorDict):  
        #     if 'observation' in state:
        obs = state['observation'].cpu().detach().numpy()
        if reduced:
            hand = obs[:1, :, :]
            remaining = np.sum(obs[1:, :, :], axis=0, keepdims=True)
            reduced_state = np.concatenate([hand, remaining], axis=0)
            reduced_state = reduced_state.sum(axis=2)
            return hash(reduced_state.tobytes())
        return hash(obs.tobytes())
        # # else try to hash the entire state
        # return hash(str(state))
    
    # def initialize_regrets(self, state):
    #     """
    #     Initialize regrets for a new state
    #     """
    #     state_key = self.get_state_key(state)
    #     if state_key not in self.cumulative_regrets:
    #         self.cumulative_regrets[state_key] = np.zeros(self.action_space_size)

    def select_action(self, state, greedy=False):
        """
        Select an action based on the regret matching policy
        
        Args:
            state: Current state (can be TensorDict)
        """
        state_key = self.get_state_key(state)
        # If no prior regrets, initialize regrets and choose based on ppo policy
        if state_key not in self.cumulative_regrets:
            # print("choosing on ppo")
            self.cumulative_regrets[state_key] = np.zeros(self.action_space_size)
            obs_dict = TensorDict({
                "observation": state['observation'],
                "legal_mask": state['legal_mask']
            }, batch_size=[])
            with torch.no_grad():
                action, _, _, _ = self.ppo_model.get_action_and_value(obs_dict)
                action = action.cpu().item()
                # print(action)
                return action
        
        self.count += 1
        # Get positive regrets
        positive_regrets = np.maximum(self.cumulative_regrets[state_key], 0)
        # Apply legal action mask if provided
        legal_mask = state['legal_mask']
        if isinstance(legal_mask, torch.Tensor):
            legal_mask = legal_mask.cpu().numpy()
        # Zero out regrets for illegal actions
        positive_regrets = positive_regrets * legal_mask
        regret_sum = np.sum(positive_regrets)
        
        # If sum of positive regrets is zero, use uniform strategy over legal actions
        if regret_sum > 0:
            policy = positive_regrets / regret_sum
        else: # regret_sum <= 0, take uniform over actions
            legal_count = np.sum(legal_mask)
            if legal_count > 0: # should be positive
                policy = np.zeros(self.action_space_size)
                policy = legal_mask / legal_count
            else: # otherwise jic, just set to uniform over everything
                policy = np.ones(self.action_space_size) / self.action_space_size

        # # Ensure the policy sums to 1 jic??
        # policy = policy / np.sum(policy)
        if not greedy:
            action = np.random.choice(self.action_space_size, p=policy)
        else:
            action = np.argmax(policy)
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
        # print("LEGAL MASK IS", sum(legal_mask))
        # count = 0

        counterfactual_values = np.full(self.action_space_size, float('-inf'))
        
        # Try each action
        for action in range(self.action_space_size):
            # Skip illegal actions
            if legal_mask[action] == 0:
                continue
            # print("action: ", count)
            # count += 1
                
            # Take the action
            input_dict = TensorDict({"action": torch.tensor([action])}, batch_size=[])
            next_state = env._step(input_dict)
            # Extract reward and done from TensorDict
            reward = next_state['reward']
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
                    
                    next_state_value = self.ppo_model.get_value(next_state_tensor).item()
                    counterfactual_values[action] = reward + next_state_value

            # Step back
            env._step_back()  

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
        if state_key not in self.cumulative_regrets:
            self.cumulative_regrets[state_key] = np.zeros(self.action_space_size)

        # Calculate the observed reward for the next state
        reward = next_state['reward']
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
                    
                next_state_value = self.ppo_model.get_value(next_state_tensor).item()
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
              eval_interval=100, eval_episodes=10, save_path=None, verbose=True, use_wandb=True):
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
            state = env._reset(None)
            episode_reward = 0
            steps = 0
            
            # Track unique states
            stats['unique_states_visited'].add(self.get_state_key(state))
            # env.mahjong_env.game.print_game_state()s
            
            done = False
            while not done and steps < max_steps_per_episode:
                # Compute true counterfactual values for all actions
                counterfactual_values = self.compute_true_counterfactual_values(env, state)

                # Select action using regret matching
                action = self.select_action(state)
                # Take action in environment
                input_dict = TensorDict({"action": torch.tensor([action])}, batch_size=[])
                next_state = env._step(input_dict)
                # env.mahjong_env.game.print_game_state()
                # Calculate the observed reward for the next state
                reward = next_state['reward']
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
            # print(self.count)

            # Record episode statistics
            stats['episode_rewards'].append(episode_reward)
            stats['regret_table_sizes'].append(len(self.cumulative_regrets))
            
            if use_wandb:
                # Log to wandb
                wandb.log({
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "regret_table_size": len(self.cumulative_regrets),
                    "unique_states": len(stats['unique_states_visited']),
                    "steps": steps
                })
            
            # Update progress bar if verbose
            if verbose:
                avg_reward = np.mean(stats['episode_rewards'][-100:]) if len(stats['episode_rewards']) >= 100 else np.mean(stats['episode_rewards'])
                progress_bar.set_description(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | States: {len(self.cumulative_regrets)}")
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(env, num_episodes=eval_episodes)
                stats['eval_rewards'].append((episode + 1, eval_reward))
                if use_wandb:
                    wandb.log({
                        "eval_episode": episode + 1,
                        "eval_reward": eval_reward,
                        "avg_100_episode_reward": np.mean(stats['episode_rewards'][-100:]) if len(stats['episode_rewards']) >= 100 else np.mean(stats['episode_rewards'])
                    })
                if verbose:
                    print(f"\nEvaluation at episode {episode+1}: Average reward: {eval_reward:.2f}\n")
                
                # Save regret table if path is provided
                if save_path:
                    self.save(f"{save_path}/rm_episode_{episode+1}.npy")
                    if use_wandb and (episode + 1) % (eval_interval * 2) == 0:
                        wandb.save(f"{save_path}/rm_episode_{episode+1}.npy")
        
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
            state = env._reset(None)
            episode_reward = 0
            done = False
            
            while not done:
                # # Extract legal mask
                # legal_mask = state['legal_mask']
                # # Select action using current policy (greedy for evaluation)
                # policy = self.get_regret_matching_policy(state, legal_mask)
                # action = np.argmax(policy)  
                
                # Greedy action selection for evaluation
                action = self.select_action(state, greedy=True)
                
                # Take action in environment
                input_dict = TensorDict({"action": torch.tensor([action])}, batch_size=[])
                next_state = env._step(input_dict)
                
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


# Train wrapper
def train_agent(args):
    # Create directories if they don't exist
    os.makedirs(args.save_path, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        run_id = f"rm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.run_name:
            run_id = f"{args.run_name}_{run_id}"
            
        wandb.init(
            project=args.wandb_project,
            name=run_id,
            config={
                "learning_rate": args.learning_rate,
                "regret_discount": args.regret_discount,
                "num_episodes": args.num_episodes,
                "max_steps": args.max_steps,
                "eval_interval": args.eval_interval,
                "eval_episodes": args.eval_episodes,
                "model_path": args.model_path,
                "device": args.device,
                "seed": args.seed
            }
        )
    # Set random seeds for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # load ppo model, initialize environment
    mahjong_config = {
        'allow_step_back': True,
        'num_players': 4,
        'seed': 0, # manually add seed here perhap
        'device': args.device
    }

    mahjong_env = MahjongEnv(mahjong_config)
    base_env = MahjongTorchEnv(mahjong_env, device=args.device)
    obs_shape = base_env.observation_spec["observation"].shape
    num_actions = base_env.action_spec['action'].n
    
    ppo = PPOActorCritic(obs_shape, num_actions)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    ppo.load_state_dict(checkpoint['model_state_dict'])
    ppo.to("cpu") 


    action_space_size = base_env.action_spec['action'].n
    rm_agent = RegretMatch(ppo, action_space_size)
    print(f"Starting training for {args.num_episodes} episodes...")
    start_time = time.time()
    stats = rm_agent.train(
        env=base_env,
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        max_steps_per_episode=args.max_steps,
        save_path=args.save_path,
        use_wandb=args.use_wandb
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate final policy
    final_reward = rm_agent.evaluate(base_env, num_episodes=args.final_eval_episodes, render=args.render)
    print(f"Final evaluation average reward: {final_reward:.2f}")
    if args.use_wandb:
        wandb.log({
                "final_eval_reward": final_reward,
                "training_time_seconds": training_time,
                "total_unique_states": stats['unique_states_visited'],
                "final_regret_table_size": len(rm_agent.cumulative_regrets)
        })
        wandb.finish()

    # Save the final model
    final_save_path = f"{args.save_path}/regret_matching_final.npy"
    rm_agent.save(final_save_path)
    print(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Regret Match Mahjong Agent")
    parser.add_argument('--model_path', type=str, default="rlcard_fork/examples/experiments/mahjong_ppo/best_model.pt", help='Path to the saved model checkpoint')
    parser.add_argument('--device', type=str, default="cpu", help='Device')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for regret updates')
    parser.add_argument('--regret_discount', type=float, default=0.95, help='Discount factor for historical regrets')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eval_interval', type=int, default=50, help='Interval to evaluate current policy')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation during training')
    parser.add_argument('--final_eval_episodes', type=int, default=20, help='Number of episodes for final evaluation')
    parser.add_argument('--save_path', type=str, default="rlcard_fork/examples/experiments/mahjong_rm", help='Path to save checkpoints')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Whether to render during final evaluation')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging')    
    parser.add_argument('--wandb_project', type=str, default="mahjong-regret-matching", help='WandB project name')
    parser.add_argument('--run_name', type=str, default="regret-matching-run", help='Name for the wandb run')


    args = parser.parse_args()

    train_agent(args)
    
