import os
import argparse
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase, EnvCreator
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data import TensorSpec
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from rlcard.envs.mahjong import MahjongEnv

from rlcard.envs.mahjong_envwrap import MahjongTorchEnv

class PPOActorCritic(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        
        # Mahjong observation: (num_players + 2, 34, 4)
        input_size = np.prod(obs_shape)
        
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def get_value(self, obs):
        features = self.features(obs)
        return self.value(features)
    
    def get_action_and_value(self, obs, action=None):
        features = self.features(obs)
        logits = self.policy(features)
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy().mean()
        value = self.value(features)
        
        return action, log_prob, entropy, value


# Self play wrapper?
class SelfPlayMahjongEnv(EnvBase):
    def __init__(self, mahjong_env, policy, device="cpu"):
        super().__init__()
        self.env = mahjong_env
        self.policy = policy
        self.device = device
        self.num_players = self.env.mahjong_env.num_players
        self.current_player_id = 0
        
        self.observation_spec = self.env.observation_spec
        self.action_spec = self.env.action_spec
        self.reward_spec = self.env.reward_spec
        self.done_spec = self.env.done_spec
        self.batch_size = self.env.batch_size
    
    def _reset(self, input_dict=None):
        state = self.env._reset(input_dict)
        self.current_player_id = 0 
        return state
    
    def _step(self, input_dict):
        action = input_dict["action"]
        
        next_state = self.env._step(TensorDict({"action": action}, batch_size=[]))
        
        if next_state["done"].item():
            return next_state
        
        while self.env.mahjong_env.game.round.current_player != self.current_player_id:
            current_obs = torch.tensor(
                self.env.mahjong_env.get_state(self.env.mahjong_env.game.round.current_player)["obs"], 
                dtype=torch.int64,
                device=self.device
            )
            
            with torch.no_grad():
                policy_action, _, _, _ = self.policy.get_action_and_value(current_obs.unsqueeze(0))
                policy_action = policy_action.cpu()
            
            next_state = self.env._step(TensorDict({"action": policy_action}, batch_size=[]))
            
            if next_state["done"].item():
                break
        
        return next_state
    
    def _set_seed(self, seed):
        self.env._set_seed(seed)


def train_ppo(args):
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
    base_env = MahjongTorchEnv(mahjong_env, device=device)
    obs_shape = base_env.observation_spec["observation"].shape
    num_actions = base_env.action_spec["action"].space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    policy = PPOActorCritic(obs_shape, num_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    env_creator = lambda: SelfPlayMahjongEnv(
        MahjongTorchEnv(MahjongEnv(mahjong_config), device=device),
        policy,
        device=device
    )
    
    collector = SyncDataCollector(
        create_env_fn=env_creator,
        create_env_kwargs={},
        policy=policy.get_action_and_value,
        frames_per_batch=args.num_steps,
        total_frames=args.total_timesteps,
        device=device,
        storing_device=device,
    )
    replay_buffer = ReplayBuffer(
        storage=TensorDict({}, batch_size=[args.num_steps]),
        sampler=None,
    )
    
    gae = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=policy.get_value,
    )

    ppo_loss = ClipPPOLoss(
        actor_network=policy.get_action_and_value,
        critic_network=policy.get_value,
        clip_epsilon=args.clip_coef,
        entropy_coef=args.ent_coef,
        value_coef=args.vf_coef,
    )
    
    num_updates = args.total_timesteps // args.batch_size
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting training for {args.total_timesteps} timesteps")
    print(f"Number of updates: {num_updates}")
    
    global_step = 0
    
    for i, tensordict_data in enumerate(collector):
        tensordict_data = tensordict_data.clone()
        
        with torch.no_grad():
            gae(tensordict_data)
        
        for epoch in range(args.update_epochs):
            indices = torch.randperm(args.num_steps)
            
            for start in range(0, args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                if end > args.num_steps:
                    end = args.num_steps
                mb_indices = indices[start:end]
                
                minibatch = tensordict_data[mb_indices]
                loss_vals = ppo_loss(minibatch)
                optimizer.zero_grad()
                loss_vals["loss"].backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
        
        if "episode_reward" in tensordict_data:
            episode_rewards.extend(tensordict_data["episode_reward"].tolist())
            episode_lengths.extend(tensordict_data["episode_length"].tolist())
            
            if len(episode_rewards) > 0:
                print(f"Update {i+1}/{num_updates}, Episode Reward: {np.mean(episode_rewards[-10:]):.2f}, Episode Length: {np.mean(episode_lengths[-10:]):.2f}")
        
        global_step += args.num_steps
        
        if (i + 1) % args.save_every == 0 or i == num_updates - 1:
            checkpoint_path = os.path.join(args.log_dir, f"model_{i+1}.pt")
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
    
    final_model_path = os.path.join(args.log_dir, "final_model_PPO.pt") # edit
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PPO Mahjong Self-Play Training")
    
    # Make config files?
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=2048, help='Number of steps to collect per batch')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--minibatch_size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs to update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_coef', type=float, default=0.2, help='PPO clip coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps to train for')
    
    parser.add_argument('--cuda', type=str, default='', help='CUDA device index, empty for CPU')
    parser.add_argument('--log_dir', type=str, default='experiments/mahjong_ppo_results/', help='Directory to save logs and models')
    parser.add_argument('--save_every', type=int, default=10, help='Save model every N updates')
    
    args = parser.parse_args()
    
    train_ppo(args)
