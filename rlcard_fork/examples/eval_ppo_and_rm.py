import os
import argparse
import torch
import numpy as np
from rlcard.envs.mahjong import MahjongEnv
from rlcard.envs.mahjong_envwrap import MahjongTorchEnv
from tensordict import TensorDict
from run_ppo import PPOActorCritic
from run_regretmatch import RegretMatch
from rlcard.agents import DQNAgent, RandomAgent
from tqdm import tqdm

def evaluate_vs_agent(args, model, env, opponent_agents, model_type="PPO", num_model=1, opp_type="DQN"):
    num_players = env.mahjong_env.num_players
    print("\n-----------------------------------------------------------------------")
    print(f"Evaluating {num_model} {model_type} against {num_players - num_model} {opp_type} agents...")
    print("-----------------------------------------------------------------------")
    num_episodes = args.num_eval_games
    num_wins = np.zeros(num_players)
    total_reward = np.zeros(num_players)

    progress_bar = tqdm(range(num_episodes))
    for episode in progress_bar:  # range(args.num_eval_games):
        state = env._reset(None)
        episode_reward = np.zeros(num_players)

        done = False
        while not done:
            current_player = env.mahjong_env.game.round.current_player
            if current_player < num_model:  # PPO or RM agent's turn
                if model_type == "PPO":
                    obs_dict = TensorDict({
                        "observation": state['observation'],
                        "legal_mask": state['legal_mask']
                    }, batch_size=[])
                    with torch.no_grad():
                        action, _, _, _ = model.get_action_and_value(obs_dict)
                        action = action.cpu().item()
                elif model_type == "RM":
                    action = model.select_action(state, greedy=True)  # can change this!!!
            else: # opponent (DQN/Random) agent's turn
                if opp_type == "DQN":
                    player_state = env.mahjong_env.get_state(current_player)
                    action, _ = opponent_agents[current_player].eval_step(player_state)
                elif opp_type == "RANDOM":
                    player_state = env.mahjong_env.get_state(current_player)
                    action = opponent_agents[current_player].step(player_state)
            
            # Take action
            tensordict = TensorDict({"action": torch.tensor([action], dtype=torch.long)}, batch_size=[])
            next_state = env._step(tensordict)
                
            state = next_state
            done = next_state["done"].item()
            reward = next_state["reward"].item()
            if not done:
                # if reward > 0:
                #     print("HERE REWARD", reward)
                episode_reward[current_player] += reward

        # Update end game payoffs
        payoffs = env.mahjong_env.get_payoffs()
        if np.max(payoffs) == 1:
            winner = np.argmax(payoffs)
            num_wins[winner] += 1
        total_reward += episode_reward
        total_reward += payoffs

        # Log progress
        progress_bar.set_description(f"Episode {episode+1}/{num_episodes}")
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{args.num_eval_games} completed")
    
    # Final eval metrics
    win_rate = num_wins / num_episodes
    avg_reward = total_reward / num_episodes
    for i in range(num_players):
        print(f"- Player {i}: Win Rate - {win_rate[i]:.4f} ({num_wins[i]}/{num_episodes}), Average Reward - {avg_reward[i]:.4f} ({total_reward[i]}/{num_episodes})")
    ppo_avg_win = np.mean(win_rate[:num_model])
    dqn_avg_win = np.mean(win_rate[num_model:])
    print(f"{model_type} Agent Average Win Rate: {ppo_avg_win}")
    print(f"{opp_type} Agent Average Win Rate: {dqn_avg_win}")

def evaluate(args):
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

    # Load PPO model
    obs_shape = env.observation_spec["observation"].shape
    num_actions = env.action_spec['action'].n
    ppo_model = PPOActorCritic(obs_shape, num_actions)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    ppo_model.load_state_dict(checkpoint['model_state_dict'])
    ppo_model.to("cpu")
    print(f"Loaded PPO model from {args.model_path}")

    # Load opponent model
    opp_type = args.opponent_agent
    num_players = env.mahjong_env.num_players
    if opp_type == "DQN":
        opponent_agents = [torch.load(args.dqn_model_path, weights_only=False, map_location=args.device) for _ in range(num_players)]
        print(f"Loaded DQN agents from {args.dqn_model_path}")
        for agent in opponent_agents:
            agent.q_estimator.device = torch.device("cpu")
        print("DQN Agent requires state shape:", opponent_agents[0].q_estimator.qnet.state_shape)
    elif opp_type == "RANDOM":
        opponent_agents = [RandomAgent(num_actions=num_actions) for _ in range(num_players)]
        print(f"Loaded RANDOM agents")

    # Evaluate model
    if args.model_to_eval == "PPO":
        evaluate_vs_agent(args, ppo_model, env, opponent_agents, num_model=1, opp_type=opp_type)
        evaluate_vs_agent(args, ppo_model, env, opponent_agents, num_model=3, opp_type=opp_type)
    if args.model_to_eval == "RM":
        action_space_size = env.action_spec['action'].n
        rm_model = RegretMatch(ppo_model, action_space_size)
        rm_model.load(args.rm_model_path)
        print(f"Loaded RM model from {args.rm_model_path}")
        evaluate_vs_agent(args, rm_model, env, opponent_agents, model_type="RM", num_model=1, opp_type=opp_type)
        evaluate_vs_agent(args, rm_model, env, opponent_agents, model_type="RM", num_model=3, opp_type=opp_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate PPO Mahjong Agent")
    parser.add_argument('--model_path', type=str,default="rlcard_fork/examples/experiments/mahjong_ppo/best_model.pt", help='Path to the saved model checkpoint')
    parser.add_argument('--model_to_eval', type=str, default="PPO", help='Model being evaluated (either PPO or RM)')
    parser.add_argument('--opponent_agent', type=str, default="DQN", help='Opponent model being evaluated against (either DQN or RANDOM)')
    parser.add_argument('--dqn_model_path', type=str, default="dqn_models/model4.pth", help='DQN model path')
    parser.add_argument('--rm_model_path', type=str, default="rlcard_fork/examples/experiments/mahjong_rm_final.npy", help='RM model path')
    
    parser.add_argument('--num_eval_games', type=int, default=100, help='Number of games to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')    
    parser.add_argument('--device', type=str, default="cpu", help='Device, default to cpu, have not tested cuda')


    args = parser.parse_args()
    
    evaluate(args)