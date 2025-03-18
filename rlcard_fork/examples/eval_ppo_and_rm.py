import os
import argparse
import torch
import numpy as np
from rlcard.envs.mahjong import MahjongEnv
from rlcard.envs.mahjong_envwrap import MahjongTorchEnv
from tensordict import TensorDict
from run_ppo import PPOActorCritic
from run_regretmatch import RegretMatch
from rlcard.agents import DQNAgent, RandomAgent, NFSPAgent
from tqdm import tqdm

def evaluate_vs_agent(args, env, model_tuple, opp_tuple, num_model):
    model_type, model = model_tuple
    opp_type, opp_agent = opp_tuple
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
            # specify if it is model's turn  OR opponent agent's turn
            current_player = env.mahjong_env.game.round.current_player
            
            cur_model_type = model_type if current_player < num_model else opp_type
            cur_model = model if current_player < num_model else opp_agent

            # get next action
            if cur_model_type == "PPO":
                obs_dict = TensorDict({"observation": state['observation'],"legal_mask": state['legal_mask']}, batch_size=[])
                with torch.no_grad():
                    action, _, _, _ = cur_model.get_action_and_value(obs_dict)
                    action = action.cpu().item()
            elif cur_model_type == "RM":
                action = cur_model.select_action(state, greedy=True)  # can change this!!!
            elif cur_model_type in ["DQN", "NFSP", "RANDOM"]:
                player_state = env.mahjong_env.get_state(current_player)
                action, _ = cur_model.eval_step(player_state)
            
            # Take action
            tensordict = TensorDict({"action": torch.tensor([action], dtype=torch.long)}, batch_size=[])
            next_state = env._step(tensordict)
                
            state = next_state
            done = next_state["done"].item()
            reward = next_state["reward"].item()
            if not done:
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
        player_type = model_type if i < num_model else opp_type
        print(f"- Player {i} ({player_type}): Win Rate - {win_rate[i]:.4f} ({num_wins[i]}/{num_episodes}), Average Reward - {avg_reward[i]:.4f} ({total_reward[i]}/{num_episodes})")
    model_avg_win = np.mean(win_rate[:num_model])
    opp_avg_win = np.mean(win_rate[num_model:])
    print(f"{model_type} Agent Average Win Rate: {model_avg_win}")
    print(f"{opp_type} Agent Average Win Rate: {opp_avg_win}")


def load(args, env, model_type):
    if model_type == "PPO" or model_type == "RM": # Load PPO model
        obs_shape = env.observation_spec["observation"].shape
        num_actions = env.action_spec['action'].n
        ppo_model = PPOActorCritic(obs_shape, num_actions)
        checkpoint = torch.load(args.ppo_model_path, map_location=args.device, weights_only=False)
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

    elif model_type == "NFSP":  # Load NFSP model
        nfsp_agent = torch.load(args.nfsp_model_path, weights_only=False, map_location='cpu')
        nfsp_agent.set_device(torch.device('cpu'))
        #nfsp_agent = NFSPAgent.from_checkpoint(checkpoint=c)
        print(f"Loaded NFSP agent from {args.nfsp_model_path}")

        return nfsp_agent
    
def evaluate(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    mahjong_config = {
        'allow_step_back': False,
        'num_players': 2,
        'seed': args.seed,
        'device': str(device)
    }
    mahjong_env = MahjongEnv(mahjong_config)
    env = MahjongTorchEnv(mahjong_env, device=device)

    model_type, opp_type = args.model_to_eval, args.opponent_agent
    model, opp_agent = load(args, env, model_type), load(args, env, opp_type)
    evaluate_vs_agent(args, env, (model_type, model), (opp_type, opp_agent), args.num_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate PPO Mahjong Agent")
    parser.add_argument('--model_to_eval', type=str, default="PPO", help='Model being evaluated (either PPO or RM)')
    parser.add_argument('--opponent_agent', type=str, default="DQN", help='Opponent model being evaluated against (either DQN or RANDOM)')
    parser.add_argument('--num_models', type=int, default=1, help="Number of models to eval against opponent agent, e.g. 1 (1v3) or 3 (3v1)")

    parser.add_argument('--ppo_model_path', type=str,default="~/cs234-project-mahjong/rlcard_fork/examples/ppo_2_explore/best_model.pt")#, help='Path to the saved model checkpoint')
    parser.add_argument('--dqn_model_path', type=str, default="~/backup_dqn/logs_mahjong/model2.pth", help='DQN model path')
    parser.add_argument('--rm_model_path', type=str, default="~/cs234-project-mahjong/rlcard_fork/examples/experiments/mahjong_rm_2_player/rm_episode_10450.npy", help='RM model path')
    parser.add_argument('--nfsp_model_path', type=str, default="~/cs234-project-mahjong/rlcard_fork/examples/experiments/leduc_holdem_nfsp_result/model.pth", help='NFSP model path')
    
    parser.add_argument('--num_eval_games', type=int, default=100, help='Number of games to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')    
    parser.add_argument('--device', type=str, default="cpu", help='Device, default to cpu, have not tested cuda')

    args = parser.parse_args()
    
    evaluate(args)