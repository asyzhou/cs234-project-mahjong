import os
import argparse
import time
import torch
import numpy as np
from rlcard.envs.mahjong import MahjongEnv
from rlcard.envs.mahjong_envwrap import MahjongTorchEnv
from tensordict import TensorDict
from run_ppo import PPOActorCritic
from run_regretmatch import RegretMatch
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.games.mahjong.utils import card_decoding_dict, card_encoding_dict


player_action_hist = []

def load_model(args, env, model_type):
    if model_type == "PPO" or model_type == "RM": # Load PPO model
        obs_shape = env.observation_spec["observation"].shape
        num_actions = env.action_spec['action'].n
        ppo_model = PPOActorCritic(obs_shape, num_actions)
        checkpoint = torch.load(args.ppo_model_path, map_location=args.device)
        ppo_model.load_state_dict(checkpoint['model_state_dict'])
        ppo_model.to("cpu")
        print(f"Loaded PPO model from {args.ppo_model_path}")

        if model_type == "RM":
            action_space_size = env.action_spec['action'].n
            rm_model = RegretMatch(ppo_model, action_space_size)
            rm_model.load(args.rm_model_path)
            print(f"Loaded RM model from {args.rm_model_path}")
            return rm_model
        else:
            return ppo_model

    elif model_type == "DQN": # Load DQN model
        dqn_agent = torch.load(args.dqn_model_path, weights_only=False, map_location=args.device) #for _ in range(num_players)]
        dqn_agent.q_estimator.device = torch.device("cpu")
        print(f"Loaded DQN agents from {args.dqn_model_path}")
        print("DQN Agent requires state shape:", dqn_agent.q_estimator.qnet.state_shape)
        return dqn_agent
        
    elif model_type == "RANDOM": # Load RANDOM model
        num_actions = env.action_spec['action'].n
        random_agent = RandomAgent(num_actions=num_actions)
        print(f"Loaded RANDOM agents")
        return random_agent

def display_game_state(state, env, human_player_id):
    """Display the current game state to the human player"""
    print("\n===== CURRENT GAME STATE =====")
    env.mahjong_env.game.print_game_state()

def get_human_action(state):
    """Prompt human player for action selection"""
    legal_actions = np.where(state['legal_mask'] == 1)[0]
    
    while True:
        try:
            choice = str(input("\nSelect action: "))
            if choice in card_encoding_dict and card_encoding_dict[choice] in legal_actions:
                player_action_hist.append(card_encoding_dict[choice])
                return card_encoding_dict[choice]
            elif choice == "history":
                print(player_action_hist)
            else:
                print("Invalid choice, please try again.")
        except ValueError:
            print("Please enter a number.")

def play_against_model(args, env, model_tuple):
    """Play a game where a human plays against model"""
    model_type, model = model_tuple
    human_player_id = args.human_player_id
    num_players = env.mahjong_env.num_players
    
    print("\n===============================")
    print(f"Playing as Player {human_player_id} against {num_players - 1} {model_type} agents...")
    print("===============================")
    
    state = env._reset(None)
    episode_reward = np.zeros(num_players)
    
    done = False
    while not done:
        # Get current player
        current_player = env.mahjong_env.game.round.current_player
        
        # Display state if it's human's turn
        if current_player == human_player_id:
            display_game_state(state, env, human_player_id)
            action = get_human_action(state)
        else:
            # Model takes action
            if model_type == "PPO":
                obs_dict = TensorDict({"observation": state['observation'], "legal_mask": state['legal_mask']}, batch_size=[])
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(obs_dict)
                    action = action.cpu().item()
            elif model_type == "RM":
                action = model.select_action(state, greedy=True)
            elif model_type in ["DQN", "NFSP", "RANDOM"]:
                player_state = env.mahjong_env.get_state(current_player)
                action, _ = model.eval_step(player_state)
                
            print(f"\nPlayer {current_player} ({model_type}) chose action: {card_decoding_dict[action]}")
            time.sleep(0.5)
        
        # Take action
        tensordict = TensorDict({"action": torch.tensor([action], dtype=torch.long)}, batch_size=[])
        next_state = env._step(tensordict)
        
        state = next_state
        done = next_state["done"].item()
        reward = next_state["reward"].item()
        if not done:
            episode_reward[current_player] += reward
    
    # Game over, show results
    payoffs = env.mahjong_env.get_payoffs()
    final_rewards = episode_reward + payoffs
    
    print("\n========== GAME OVER ==========")
    winner = np.argmax(payoffs)
    print(f"Player {winner} wins!")
    
    for i in range(num_players):
        player_type = "Human" if i == human_player_id else model_type
        print(f"Player {i} ({player_type}): Final Reward = {final_rewards[i]:.4f}")

def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    mahjong_config = {
        'allow_step_back': False,
        'num_players': 4,
        'seed': args.seed,
        'device': str(device)
    }
    
    mahjong_env = MahjongEnv(mahjong_config)
    env = MahjongTorchEnv(mahjong_env, device=device)
    
    model_type = args.opponent_agent
    model = load_model(args, env, model_type)
    play_against_model(args, env, (model_type, model))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Play Mahjong Against AI")
    parser.add_argument('--opponent_agent', type=str, default="PPO", help='Model to play against (PPO, RM, DQN, or RANDOM)')
    parser.add_argument('--human_player_id', type=int, default=0, help='Player ID for the human (0-3)')
    
    parser.add_argument('--ppo_model_path', type=str, default="rlcard_fork/examples/experiments/mahjong_ppo/best_model.pt", help='Path to the PPO model checkpoint')
    parser.add_argument('--dqn_model_path', type=str, default="rlcard_fork/examples/experiments/mahjong_dqn/model4.pth", help='DQN model path')
    parser.add_argument('--rm_model_path', type=str, default="rlcard_fork/examples/experiments/mahjong_rm_final.npy", help='RM model path')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    main(args)