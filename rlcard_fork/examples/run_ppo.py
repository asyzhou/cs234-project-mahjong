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
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential
from torch.distributions import Categorical

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
    
    def get_value(self, observation):
        print("GET CALUSFSF")
        features = self.features(observation)
        return self.value(features)
    
    def get_action_and_value(self, tensordict, action=None):
        print("GETACIONTO VALUE")
        observation = tensordict["observation"]        
        legal_mask = tensordict["legal_mask"]
        print(observation.shape)
        print(observation.flatten().shape)
        features = self.features(observation)
        logits = self.policy(features)
        logits[~legal_mask] = float('-inf')
        
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
        print('wrapper device init,', self.device)
    
    def _reset(self, input_dict=None):
        print('CALLING RESET_')
        state = self.env._reset(input_dict)
        self.current_player_id = 0 
        return state.to(self.device)
    
    def _step(self, input_dict):
        print("CALLIGN _STEP")
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
            legal_actions = self.env.mahjong_env.get_state(self.env.mahjong_env.game.round.current_player)["legal_actions"]
            
            with torch.no_grad():
                policy_action, _, _, _ = self.policy.get_action_and_value(current_obs.unsqueeze(0), legal_actions)
                policy_action = policy_action.cpu()
            
            next_state = self.env._step(TensorDict({"action": policy_action}, batch_size=[]))
            
            if next_state["done"].item():
                break
        return next_state.to(self.device)
    
    def _set_seed(self, seed):
        self.env._set_seed(seed)

    '''@property
    def _has_dynamic_specs(self):
        return self.env._has_dynamic_specs
    
    @property
    def action_keys(self):
        return self.env.action_keys'''


def train_ppo(args):
    device = torch.device("cuda:"+str(args.cuda))
    print("device,", device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    mahjong_config = {
        'allow_step_back': False,
        'num_players': 4,
        'seed': args.seed,
        'device': 'cuda:7'
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
    env_creator = lambda: SelfPlayMahjongEnv(
        MahjongTorchEnv(MahjongEnv(mahjong_config), device=device),
        policy,
        device=device
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.num_steps),
        sampler=None,
    )
    
    gae = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=policy.get_value,
    )
    critic_module = TensorDictModule(
        module=policy.value,  
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    feature_module = TensorDictModule(
        module=policy.features,        
        in_keys=["observation"],       
        out_keys=["hidden"],          
    )

    logits_module = TensorDictModule(
        module=policy.policy,          
        in_keys=["hidden"],
        out_keys=["logits"],          
    )

    class MaskLogitsModule(nn.Module):
        def forward(self, tensordict):
            logits = tensordict["logits"]
            legal_mask = tensordict["legal_mask"]
            logits[~legal_mask] = float("-inf")
            tensordict.set("logits", logits)
            return tensordict

    mask_module = TensorDictModule(
        module=MaskLogitsModule(),
        in_keys=["logits", "legal_mask"],  
        out_keys=["logits"],                # override logits
    )

    
    prob_module = ProbabilisticTensorDictModule(
        in_keys=["logits"],            
        out_keys=["action", "log_prob"],
        distribution_class=Categorical, 
        distribution_kwargs={}
    )

    actor_network = ProbabilisticTensorDictSequential(
        TensorDictModule(
            module=policy.features,
            in_keys=["observation"],
            out_keys=["hidden"],
        ),
        feature_module,
        logits_module,
        mask_module,
        prob_module
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
    )
    
    num_updates = args.total_timesteps // args.batch_size
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    print("Starting training for" + str(args.total_timesteps) + " timesteps")
    print("Number of updates"+ str(num_updates))
    
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
