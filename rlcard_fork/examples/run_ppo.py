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
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, TensorDictSequential
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor
import wandb

# class BatchedPolicyWrapper(nn.Module):
#     def __init__(self, policy_net, transformer):
#         super().__init__()
#         self.policy_net = policy_net
#         self.trans = transformer

#     def forward(self, observation):
#         if observation.dim() == 3:  # Single observation case (6, 36, 4)
#             observation = observation.unsqueeze(0)  # Convert to (1, 6, 36, 4)
#         observation = observation.flatten(start_dim=1)
#         return self.policy_net(self.trans(observation))

class BatchedPolicyWrapper(nn.Module):
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net

    def forward(self, observation):
        if observation.dim() == 3:  # Single observation case (6, 36, 4)
            observation = observation.unsqueeze(0)  # Convert to (1, 6, 36, 4)
        return self.policy_net(observation)

class BatchedValueWrapper(nn.Module):
    def __init__(self, value_net, transformer):
        super().__init__()
        self.value_net = value_net
        self.trans = transformer

    def forward(self, observation):
        if observation.dim() == 3:  # Single observation case (6, 36, 4)
            observation = observation.unsqueeze(0)  # Convert to (1, 6, 36, 4)
        observation = observation.flatten(start_dim=1)
        return self.value_net(self.trans(observation))

# sparse observation and no reward in the beginning
# policy - mlp to predict action - which tile to throw away and which to take
# Maybe too hard in the beginning for model to learn
# 150 is too long trajectory - a lot of actions are redundant/not correct at all
# really noisy trajectories can impact ppo training
# insert some intermediate reward signal - check number of pairs are matched to give partial rewards
# Other way - try to create hand-craft data to see if its data or model problem
# See if ppo on hand-crafted games to see if the data is wrong
# self-play - need some searching - add heuristic in action planning/policy
# 

class PPOActorCritic(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        
        # Mahjong observation: (num_players + 2, 34, 4)
        input_size = np.prod(obs_shape)
        
        # self.features = nn.Sequential(
        #     nn.Flatten(start_dim=0, end_dim=-1),
        #     nn.Linear(input_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        # )
        
        self.raw_policy = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size, 512),  # Increase from 512
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        self.policy = BatchedPolicyWrapper(self.raw_policy)
        
        self.value = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,  256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def get_value(self, observation):
        # features = self.features(observation)
        if observation.dim() == 3: 
            observation = observation.unsqueeze(0)
        return self.value(observation)
    
    def get_action_and_value(self, observation, action=None):
        if isinstance(observation, dict) or isinstance(observation, TensorDict):
            obs_tensor = observation["observation"]
            legal_mask = observation.get("legal_mask", None)
        else:
            obs_tensor = observation
            legal_mask = None
        
        # features = self.features(obs_tensor)
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)

        logits = self.policy(obs_tensor)
        
        if legal_mask is not None:
            if not legal_mask.dtype == torch.bool:
                legal_mask = legal_mask.bool()
            logits = logits.clone()
            legal_mask = legal_mask.unsqueeze(0)
            logits[~legal_mask] = float('-inf')
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy().mean()

        value = self.value(obs_tensor)
        
        return action, log_prob, entropy, value


# Self play wrapper
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
        print('wrapper device init,', self.device)
    
    def _reset(self, input_dict=None):
        state = self.env._reset(input_dict)
        self.current_player_id = 0 
        return state.to(self.device)
    
    def _step(self, input_dict):
        action = input_dict["action"]
        
        next_state = self.env._step(TensorDict({"action": action}, batch_size=[]))
        
        if next_state["done"].item():
            return next_state
        
        while self.env.mahjong_env.game.round.current_player != self.current_player_id:
            current_player_id = self.env.mahjong_env.game.round.current_player
            player_state = self.env.mahjong_env.get_state(current_player_id)
            
            current_obs = torch.tensor(
                player_state["obs"], 
                dtype=torch.float,  
                device=self.device
            )
            
            legal_actions = player_state["legal_actions"]
            legal_mask = torch.zeros(self.env.action_spec["action"].n, dtype=torch.bool, device=self.device)
            for act_id in legal_actions:
                legal_mask[act_id] = True
                
            obs_dict = TensorDict({
                "observation": current_obs,
                "legal_mask": legal_mask
            }, batch_size=[])
            
            
            with torch.no_grad():
                policy_action, _, _, _ = self.policy.get_action_and_value(obs_dict)
                policy_action = policy_action.cpu()
            
            next_state = self.env._step(TensorDict({"action": policy_action}, batch_size=[]))
            
            if next_state["done"].item():
                break
        
        return next_state.to(self.device)
    
    def _set_seed(self, seed):
        self.env._set_seed(seed)



def train_ppo(args):
    wandb.init(
        project="ppo-mahjong",  
        config=vars(args) 
    )
    device = torch.device("cuda:"+str(args.cuda) if args.cuda else "cpu")

    print("device,", device)
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
    base_env = MahjongTorchEnv(mahjong_env, device=device)
    obs_shape = base_env.observation_spec["observation"].shape
    print(base_env.action_spec)
    num_actions = base_env.action_spec['action'].n
    
    print("Observation shape,",  obs_shape)
    print("Number of actions,", num_actions)
    policy = PPOActorCritic(obs_shape, num_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt, map_location=device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        if 'best_reward' in checkpoint:
            best_reward = checkpoint['best_reward']
        print(f"Loaded checkpoint from {args.load_ckpt}, starting at global_step={global_step}")
        for param_group in optimizer.param_groups:
            print(f"Loaded optimizer state: learning rate = {param_group['lr']}, weight decay = {param_group.get('weight_decay', 0)}")
        policy.train()
        for module in policy.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.Dropout):
                module.train()

    env_creator = lambda: SelfPlayMahjongEnv(
        MahjongTorchEnv(MahjongEnv(mahjong_config), device=device),
        policy,
        device=device
    )
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.num_steps),
        sampler=None,
    )

    value_module = TensorDictModule(
        policy.value, in_keys=["observation"], out_keys=["state_value"]
    )
    
    gae = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=value_module,
    )
    
    class CriticNetwork(nn.Module):
        def __init__(self, value): # features, 
            super().__init__()
            # self.features = features
            self.value = value
        
        def forward(self, obs):
            value = self.value(obs)
            return value
    
    critic_module = TensorDictModule(
        module=CriticNetwork(policy.value), # policy.features,
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    # feature_module = TensorDictModule(
    #     module=policy.features,
    #     in_keys=["observation"],
    #     out_keys=["hidden"],
    # )

    logits_module = TensorDictModule(
        module=policy.policy,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    class SimpleMaskModule(nn.Module):
        def forward(self, logits, legal_mask):
            if legal_mask.dtype != torch.bool:
                legal_mask = legal_mask.bool()
            if legal_mask.dim() == 1:  
                legal_mask = legal_mask.unsqueeze(0)
            
            masked_logits = logits.clone()
            masked_logits[~legal_mask] = float('-inf')
            return masked_logits
    
    mask_module = TensorDictModule(
        module=SimpleMaskModule(),
        in_keys=["logits", "legal_mask"],
        out_keys=["logits"],
    )

    actor_module = TensorDictSequential(
        logits_module,
        mask_module,
    )
    
    
    actor_network = ProbabilisticActor(
        module=actor_module,
        distribution_class=Categorical,
        in_keys=["logits"],
        out_keys=["action"],
        return_log_prob=True,
    )

    collector = SyncDataCollector(
        create_env_fn=env_creator,
        create_env_kwargs={},
        policy=actor_network,
        frames_per_batch=args.num_steps,
        total_frames=args.total_timesteps,
        device=device,
        storing_device=device,
    )

    ppo_loss = ClipPPOLoss(
        actor_network=actor_network,
        critic_network=critic_module,
        clip_epsilon=args.clip_coef,
        entropy_coef=args.ent_coef,
        value_coef=args.vf_coef,
        entropy_bonus=True
    )
    
    num_updates = args.total_timesteps // args.batch_size
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting training for {args.total_timesteps} timesteps")
    print(f"Number of updates: {num_updates}")
    
    global_step = 0
    best_reward = float('-inf')
    
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
                #loss_vals.backward()
                total_loss = (loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"])
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                wandb.log({
                    "loss_objective": loss_vals["loss_objective"].item(),
                    "entropy": loss_vals["entropy"].item(),
                    "kl_approx": loss_vals["kl_approx"].item(),  
                    "loss_critic": loss_vals["loss_critic"].item(),
                    "loss_entropy": loss_vals["loss_entropy"].item(),
                    "global_step": global_step
                })
        
        total_reward = tensordict_data['next']['reward'].sum()
        wandb.log({
                "total_episode_reward": total_reward
            })
    
        global_step += args.num_steps

        if (i+1) > 100 and total_reward > best_reward:
            best_reward = total_reward
            best_model_path = os.path.join(args.log_dir, "best_model.pt")
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_reward': best_reward
            }, best_model_path)
            print(f"Best model saved with reward {best_reward} at {best_model_path}")
        
        if (i + 1) % args.save_every == 0 or i == num_updates - 1:
            checkpoint_path = os.path.join(args.log_dir, f"model_{i+1}.pt")
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
    
    final_model_path = os.path.join(args.log_dir, "final_model_PPO.pt")
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PPO Mahjong Self-Play Training")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=2000, help='Number of steps to collect per batch')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--minibatch_size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs to update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_coef', type=float, default=0.2, help='PPO clip coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.02, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps to train for')
    
    parser.add_argument('--cuda', type=str, default='', help='CUDA device index, empty for CPU')
    parser.add_argument('--log_dir', type=str, default='experiments/mahjong_ppo_results/', help='Directory to save logs and models')
    parser.add_argument('--save_every', type=int, default=300, help='Save model every N updates')
    parser.add_argument('--load_ckpt', type=str, default=None)#'experiments/mahjong_ppo/model_1400.pt',
                        #help='Path to a .pt checkpoint fidle to resume training')
    
    '''parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=5000, help='Number of steps to collect per batch')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--minibatch_size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--update_epochs', type=int, default=7, help='Number of epochs to update')
    parser.add_argument('--gamma', type=float, default=0.97, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.9, help='GAE lambda parameter')
    parser.add_argument('--clip_coef', type=float, default=0.1, help='PPO clip coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.07, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.25, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps to train for')
    
    parser.add_argument('--cuda', type=str, default='', help='CUDA device index, empty for CPU')
    parser.add_argument('--log_dir', type=str, default='experiments/mahjong_ppo_results/', help='Directory to save logs and models')
    parser.add_argument('--save_every', type=int, default=100, help='Save model every N updates')'''
    
    args = parser.parse_args()
    
    train_ppo(args)