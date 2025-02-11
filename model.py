import numpy as np
from typing import Union, List, Tuple, Dict
import warnings
from rewardgym.utils import check_seed


class ValenceHybridAgent:
    """
    A reinforcement learning agent implementing a Valence-based Q-learning algorithm.
    The agent maintains state-action values (Q-values) and updates them using different
    learning rates for positive and negative temporal differences.
    """

    def __init__(
        self,
        alpha_mf_pos: float,
        alpha_mf_neg: float,
        alpha_mb: float,
        discount_factor: float,
        hybrid: float,
        temperature: float,
        action_space: int = 2,
        state_space: int = 2,
        seed: Union[int, np.random.Generator] = 1000,
        use_fixed = True,
        eligiblity_decay: float = 1,
        graph: Dict =None
    ):

        self.q_mf = np.zeros((state_space, action_space))
        self.q_mb = np.zeros((state_space, action_space))
        self.t_values = np.zeros((state_space, action_space, state_space))
        self.t_values = self.create_t_values_from_graph(graph=graph,
                                                   t_values=self.t_values,
                                                   use_fixed=use_fixed)
        self.eligibility = np.zeros_like(self.q_mf)

        self.q_values_hybrid = np.zeros_like(self.q_mf)


        self.terminal_states = []

        self.n_states = state_space
        self.n_actions = action_space
        self.lr_neg = alpha_mf_neg
        self.lr_pos = alpha_mf_pos
        self.lr_mb = alpha_mb
        self.gamma = discount_factor
        self.eligiblity_decay = eligiblity_decay
        self.hybrid=hybrid

        self.temperature = temperature

        self.rng = check_seed(seed)

    def get_action(self, obs: Tuple[int, int, bool], avail_actions: List = None) -> int:

        prob = self.get_probs(obs, avail_actions)
        a = self.rng.choice(np.arange(len(prob)), p=prob)

        return a

    def get_probs(self, obs: Tuple[int, int, bool], avail_actions: List = None):

        prob = np.zeros_like(self.q_mf[obs])

        if avail_actions is None:
            avail_actions = np.arange(len(self.q_mf[obs]))

        qval = self.q_values_hybrid[obs][avail_actions]

        # qval = qval - np.mean(qval)
        qs = np.exp(qval * self.temperature)

        if any(~np.isfinite(qs)):
            warnings.warn("Overflow in softmax, replacing with max / min value.")
            qs[np.isposinf(qs)] = np.finfo(float).max
            qs[np.isneginf(qs)] = np.finfo(float).min

        prob[avail_actions] = qs / np.sum(qs)

        return prob

    def update(
        self,
        obs: Tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Tuple[int, int, bool],
        **kwargs
    ):
        self.eligibility[obs, action] += 1

        # Learning process for Qlearning
        if not terminated:
            rpe_sarsa = reward + self.gamma * np.max(self.q_mf[next_obs, :]) - self.q_mf[obs, action]
        else:
            rpe_sarsa = reward - self.q_mf[obs, action]

        if rpe_sarsa <= 0:
            self.q_mf += self.lr_neg * rpe_sarsa * self.eligiblity_decay * self.eligibility
        else:
            self.q_mf += self.lr_pos * rpe_sarsa * self.eligiblity_decay * self.eligibility

        rpe_forward = 1 - self.t_values[obs, action, next_obs]

        for no in range(self.t_values.shape[-1]):
            if no == next_obs:
                self.t_values[obs, action, no] += self.lr_mb * rpe_forward
            else:
                self.t_values[obs, action, no] *= (1 - self.lr_mb)

        for tobs in range(self.t_values.shape[0]):

            for tact in range(self.t_values.shape[1]):
                if tobs in self.terminal_states:
                    self.q_mb[tobs, tact] = self.q_mf[tobs, tact]
                else:
                    qval_mb = 0
                    for no in range(self.t_values.shape[2]):
                        qval_mb += self.t_values[tobs, tact, no] * (np.max(self.q_mf[no, :]))

                    self.q_mb[tobs, tact] = qval_mb

        if terminated:
            self.eligibility *= 0

            if obs not in self.terminal_states:
                self.terminal_states.append(obs)

        else:
            self.eligibility *= self.gamma * self.eligiblity_decay

        self.q_values_hybrid = self.hybrid * self.q_mb + (1 - self.hybrid) * self.q_mf

        return self.q_mf, self.q_mb


    @staticmethod
    def create_t_values_from_graph(graph, t_values, use_fixed=True):

        if graph is not None:
            for k in graph.keys():
                actions = list(graph[k].keys())

                for a in actions:
                    loc = graph[k][a]

                    if isinstance(graph[k][a], tuple):
                        prob = graph[k][a][1]
                        loc = graph[k][a][0]
                    else:
                        prob = None

                    loc = [loc] if isinstance(loc, int) else loc
                    ln = len(loc)

                    if use_fixed and prob is not None:
                        for n, j in enumerate(loc):
                            if n == 0:
                                t_values[k, a, j] = prob
                            else:
                                t_values[k, a, j] = (1 - prob) / max([1, ln - 1])
                    else:
                        for j in loc:
                            t_values[k, a, j] = 1 / max([1, ln])

        return t_values


class HybridAgent(ValenceHybridAgent):

    def __init__(
        self,
        alpha_mf: float,
        alpha_mb: float,
        discount_factor: float,
        hybrid: float,
        temperature: float,
        action_space: int = 2,
        state_space: int = 2,
        seed: Union[int, np.random.Generator] = 1000,
        use_fixed = True,
        eligiblity_decay: float = 1,
        graph: Dict =None
    ):
        super().__init__(
            alpha_mf_neg=alpha_mf,
            alpha_mf_pos=alpha_mf,
            alpha_mb=alpha_mb,
            hybrid=hybrid,
            graph=graph,
            use_fixed=use_fixed,
            temperature=temperature,
            discount_factor=discount_factor,
            eligiblity_decay=eligiblity_decay,
            action_space=action_space,
            state_space=state_space,
            seed=seed)

class RandomAgent(ValenceHybridAgent):

    def __init__(
        self,
        action_space: int = 2,
        state_space: int = 2,
        seed: Union[int, np.random.Generator] = 1000) -> None:

        self.action_space = action_space
        self.state_space = state_space
        self.rng = check_seed(seed)

    def update(self, *args, **kwargs):
        return None

    def get_probs(self, obs, avail_actions=None):
        if avail_actions is None:
            avail_actions = np.arange(self.action_space)

        action_probs = np.zeros(self.action_space)
        action_probs[avail_actions] = 1 / len(avail_actions)

        prob = action_probs

        return prob
