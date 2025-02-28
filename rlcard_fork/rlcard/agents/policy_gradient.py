'''
Policy gradient -- modified from hw assign2
'''

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as ptd
import gym
import os
from general import get_logger, Progbar, export_plot
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy


class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, env, config, seed, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module

        You do not need to implement anything in this function. However,
        you will need to use self.discrete, self.observation_dim,
        self.action_dim, and self.lr in other methods.
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.seed = seed

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.env.seed(self.seed)

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.lr = self.config.learning_rate

        self.init_policy()

        if config.use_baseline:
            self.baseline_network = BaselineNetwork(env, config)

    def init_policy(self):
        """
        Please do the following:
        1. Create a network using build_mlp. It should map vectors of size
           self.observation_dim to vectors of size self.action_dim, and use
           the number of layers and layer size from self.config
        2. If self.discrete = True (meaning that the actions are discrete, i.e.
           from the set {0, 1, ..., N-1} where N is the number of actions),
           instantiate a CategoricalPolicy.
           If self.discrete = False (meaning that the actions are continuous,
           i.e. elements of R^d where d is the dimension), instantiate a
           GaussianPolicy. Either way, assign the policy to self.policy
        3. Create an Adam optimizer for the policy, with learning rate self.lr
           Note that the policy is an instance of (a subclass of) nn.Module, so
           you can call the parameters() method to get its parameters.
        """
        #######################################################
        #########   YOUR CODE HERE - 8-12 lines.   ############
        self.network = build_mlp(self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size)
        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        #######################################################
        #########          END YOUR CODE.          ############

    def init_averages(self):
        """
        You don't have to change or use anything here.
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, env, num_episodes=None):
        """
        Removed but was in the original code, see same function in PPOAgent
        """
        raise NotImplementedError


    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path

        TODO: compute and return G_t for each timestep. Use self.config.gamma.
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            #######################################################
            #########   YOUR CODE HERE - 5-10 lines.   ############
            returns = rewards.copy()
            for i in range(len(returns) - 1):
                returns[len(returns) - i - 2] += self.config.gamma * returns[len(returns) - i - 1]
            #######################################################
            #########          END YOUR CODE.          ############
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns

    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]

        TODO:
        Normalize the advantages so that they have a mean of 0 and standard
        deviation of 1. Put the result in a variable called
        normalized_advantages (which will be returned).

        Note:
        This function is called only if self.config.normalize_advantage is True.
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        normalized_advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        #######################################################
        #########          END YOUR CODE.          ############
        return normalized_advantages

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(self, observations, actions, advantages):
        """
        Removed but was in the original code, see same function in PPOAgent
        """
        raise NotImplementedError


    def train(self):
        """
        Removed but was in the original code, see same function in PPOAgent
        """
        raise NotImplementedError


    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True
        )
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()



class CategoricalPolicy(nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        distribution = ptd.Categorical(logits=self.network(observations))
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############
        action_dist = self.action_distribution(observations)
        sampled_actions = action_dist.sample()
        log_probs = action_dist.log_prob(sampled_actions).detach().numpy()
        sampled_actions = sampled_actions.detach().numpy()
        #######################################################
        #########          END YOUR CODE.          ############
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions
