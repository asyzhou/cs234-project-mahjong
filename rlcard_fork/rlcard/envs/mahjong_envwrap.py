import torch
from torch import Tensor
from typing import Dict
from torchrl.envs import EnvBase
from rlcard.envs.mahjong import MahjongEnv
from torchrl.data import TensorSpec, Composite, Bounded, DiscreteTensorSpec
from tensordict import TensorDict

class MahjongTorchEnv:
    def __init__(self, mahjong_env, device="cpu"):
        self.device = device
        # Store the underlying environment
        self.mahjong_env = mahjong_env
        self.batch_size = torch.Size([])
        # You should define or set up your observation and action specs. 
        # For example, if the observation is a 1D vector of length N:
        obs_shape = (self.mahjong_env.num_players + 2, 34, 4)
        self.observation_spec = Composite({
            "observation": Bounded(
                low=0,
                high=1,
                shape=obs_shape,
                dtype=torch.float,
                device=self.device,
            ),
            "legal_mask": DiscreteTensorSpec(
                n=2,  # Binary mask (0 or 1)
                shape=(34,),  # Match your action space size
                dtype=torch.bool,
                device=self.device
            )
        })
        # If your actions are discrete with M possible actions:
        self.action_spec = Composite({
            "action": DiscreteTensorSpec(
                n=34,
                shape=(1,),
                dtype=torch.long
            )
        })        
        # Reward specs, etc. (this is optional, but recommended)
        # By default you can set them to unbounded if uncertain:
        self.reward_spec = Composite({
            "reward": Bounded(
                low=-1,
                high=1,
                shape=(1,),
                dtype=torch.float32
            )
        })
        
        # Done spec is a boolean flag in TorchRL:
        self.done_spec = Composite({
            "done": DiscreteTensorSpec(
                n=2,
                shape=(1,),
                dtype=torch.bool
            )
        })

        self.current_player_id = 0
    
    def _reset(self, inputDict): # -> Dict[str, Tensor]:
        """
        _reset is where you call your environment's reset logic
        and prepare the initial state for TorchRL.
        Returns a dictionary containing at least "observation".
        """
        obs, _ = self.mahjong_env.reset()
        obs_t = torch.tensor(obs["obs"], dtype=torch.float)
        stupidass_dict = obs["legal_actions"]
        legal_mask = torch.zeros(self.action_spec["action"].n, dtype=torch.bool)
        for act_id in stupidass_dict:
            legal_mask[act_id] = True
        #obs_t = obs_t.unsqueeze(0)
        return TensorDict({"observation": obs_t, 'legal_mask': legal_mask})
    
    def _step(self, inputDict: TensorDict): # -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
        """
        _step runs one step in the environment using the provided action.
        In TorchRL, _step must return four dictionaries (or specs):
        
        next_observation: at least {"observation": <Tensor>}
        reward: at least {"reward": <Tensor>}
        done: at least {"done": <Tensor>}
        info: dictionary with additional data about the step
        """
        # Convert Torch action to something your MahjongEnv expects:
        # if action is discrete, action.item() might give the correct integer action
        action_val = inputDict["action"].item()

        cur_player = self.mahjong_env.game.round.current_player
        cur_hand = self.mahjong_env.game.players[cur_player].hand.copy()
        # print("In mahjong env", [card.get_str() for card in cur_hand])

        next_state_dict, _ = self.mahjong_env.step(action_val)
        
        obs_t = torch.tensor(next_state_dict['obs'], dtype=torch.float)  # look at _extract state function in mahjong
        stupidass_dict = next_state_dict["legal_actions"]
        legal_mask = torch.zeros(self.action_spec["action"].n, dtype=torch.bool)
        for act_id in stupidass_dict:
            legal_mask[act_id] = True
        done = self.mahjong_env.is_over()
        done_t = torch.tensor([done], dtype=torch.bool)
        reward = torch.tensor([0], dtype=torch.float32)
        if done:
            payoffs = self.mahjong_env.get_payoffs()
            reward = payoffs[cur_player]   
        if not done:
            # print("handcrafted for player: ", cur_player)
            reward = self.mahjong_env.game.judger.get_handcrafted_reward(action_val, cur_hand)
            # print("reward is: ", reward)

        reward_t = torch.tensor([reward], dtype=torch.float32)
        # if reward > 0:
        #     print("HERE REWARD_T", reward_t)
        '''
        TODO: 
        - check for multi agent: how do we return rewards/actions/observations
        - how to return losing (-1) rewards for other agents, if trajectories end once someone wins 
        '''
        result = TensorDict(
            {
                "done": done_t,
                "reward": reward_t,
                "observation": obs_t,
                "legal_mask": legal_mask,
        }, batch_size=self.batch_size)
        
        return result
    
    def _step_back(self): # -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
        """
        _step_back takes one step back through the game, user must make sure to allow step back in the orig mahjong env
        """
        state_dict, player_id = self.mahjong_env.step_back()
        cur_player = self.mahjong_env.game.round.current_player
        if cur_player != player_id:
            print("SOMETHING IS WRONG")
            return
        
        obs_t = torch.tensor(state_dict['obs'], dtype=torch.float)  # look at _extract state function in mahjong
        legal_actions = state_dict["legal_actions"]
        legal_mask = torch.zeros(self.action_spec["action"].n, dtype=torch.bool)
        for act_id in legal_actions:
            legal_mask[act_id] = True
        done = False
        done_t = torch.tensor([done], dtype=torch.bool)
        reward = torch.tensor([0], dtype=torch.float32)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        result = TensorDict(
            {
                "done": done_t,
                "reward": reward_t,
                "observation": obs_t,
                "legal_mask": legal_mask,
        }, batch_size=self.batch_size)
        
        return result
    
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

# Usage example:
if __name__ == "__main__":
    # Suppose you have an existing MahjongEnv:
    config = {
        'allow_step_back': False,
        'seed': 42,
    }
    mahjong_env = MahjongEnv(config)
    print(mahjong_env)
    env = MahjongTorchEnv(mahjong_env)
    
    # Example usage
    tens_action = torch.tensor([1], dtype=torch.long)  # e.g., first action
    state = env.reset()  # TorchRL calls env.reset()

    action_td = TensorDict({"action": tens_action}, batch_size=[])

    next_td = env.step(action_td)
    next_obs = next_td["observation"]
    reward   = next_td["reward"]
    done     = next_td["done"]
    info     = next_td.get("info", {})
    
    print("Initial State:", state)
    print("Next Observation:", next_obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
