import os
import argparse
import torch
import numpy as np
from rlcard.envs.mahjong import MahjongEnv
from mahjong_torch_env import MahjongTorchEnv
from tensordict import TensorDict
from run_ppo import PPOActorCritic

def evaluate_agent(args):
    device = torch.device(f"cuda:{args.cuda}" if args.cuda != "" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    

    mahjong_config = {
        'allow_step_back': False,
        'seed': args.seed,
    }
    mahjong_env = MahjongEnv(mahjong_config)
    env = MahjongTorchEnv(mahjong_env, device=device)
    
    
    obs_shape = env.observation_spec["observation"].shape
    num_actions = env.action_spec["action"].space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    policy = PPOActorCritic(obs_shape, num_actions).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()  
    
    print(f"Loaded model from {args.model_path}")
    
    from rlcard.agents import RandomAgent
    random_agents = [RandomAgent(num_actions=env.mahjong_env.num_actions) for _ in range(env.mahjong_env.num_players)]
    
    num_players = 2
    print("Evaluating against random agents...")
    
    num_wins = 0
    total_reward = 0
    
    for episode in range(args.num_eval_games):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            current_player = env.mahjong_env.game.round.current_player
            
            if current_player == 0:  
                obs = state["observation"].to(device)
                
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(obs.unsqueeze(0))
                    action = action.cpu()
                
                tensordict = TensorDict({"action": action}, batch_size=[])
                next_state = env.step(tensordict)
                
                state = next_state
                done = next_state["done"].item()
                reward = next_state["reward"].item()
                episode_reward += reward
            else:  
                
                player_state = env.mahjong_env.get_state(current_player)
                
                action = random_agents[current_player].step(player_state)
                
                action_tensor = torch.tensor([action], dtype=torch.long)
                tensordict = TensorDict({"action": action_tensor}, batch_size=[])
                next_state = env.step(tensordict)
                
                state = next_state
                done = next_state["done"].item()
        
        payoffs = env.mahjong_env.get_payoffs()
        if payoffs[0] > 0:
            num_wins += 1
        
        total_reward += episode_reward
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{args.num_eval_games} completed")

    win_rate = num_wins / args.num_eval_games
    avg_reward = total_reward / args.num_eval_games
    
    print(f"Evaluation Results:")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Total Wins: {num_wins}/{args.num_eval_games}")

    
    if args.eval_self_play:
        print("\nEvaluating in self-play (all agents use the same policy)...")
        
        position_wins = [0] * num_players
        total_games = args.num_eval_games
        
        for episode in range(total_games):
            env.mahjong_env.reset()
            done = False
            
            while not done:
                current_player = env.mahjong_env.game.round.current_player
                
                state = env.mahjong_env.get_state(current_player)
                
                obs = torch.tensor(state["obs"], dtype=torch.int64, device=device)
                
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(obs.unsqueeze(0))
                    action = action.item()
                
                
                _, _ = env.mahjong_env.step(action)
                
                done = env.mahjong_env.is_over()
            
            payoffs = env.mahjong_env.get_payoffs()
            winner = np.argmax(payoffs)
            position_wins[winner] += 1
            
            if (episode + 1) % 100 == 0:
                print(f"Self-play Episode {episode+1}/{total_games} completed")
        
        print("\nSelf-Play Results:")
        for i in range(num_players):
            win_rate = position_wins[i] / total_games
            print(f"Player {i} Win Rate: {win_rate:.4f} ({position_wins[i]}/{total_games})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate PPO Mahjong Agent")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--num_eval_games', type=int, default=1000, help='Number of games to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', type=str, default='', help='CUDA device index, empty for CPU')
    parser.add_argument('--eval_self_play', action='store_true', help='Evaluate in self-play mode')
    
    args = parser.parse_args()
    
    evaluate_agent(args)