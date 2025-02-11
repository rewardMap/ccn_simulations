# Helper functions for RL
from rewardgym import get_configs
import numpy as np

def run_episode(env, agent, config, n):

    episode = []
    
    obs, info = env.reset(agent_location=0, condition=config)

    done = False

    while not done:

        old_info = info    
        action = agent.get_action(obs, info["avail-actions"])

        next_obs, reward, terminated, truncated, info = env.step(
            action, step_reward=env.name=='two-step'
        )

        episode.append((action, reward, obs, next_obs, terminated, old_info["avail-actions"]))

        agent.update(obs, action, reward, terminated, next_obs, info=info)
        
        done = terminated or truncated
        obs = next_obs

    return episode


def run_episodes(env, agent, seed_int):

    settings = get_configs(env.name)(seed_int)
    actions = []
    rewards = [] 
    obs0 = []
    obs1 = []
    terminated = []
    avail_actions = []
    n_episodes = settings["ntrials"]

    for n in range(n_episodes):

        episode_data = run_episode(env, agent, config=settings["condition_dict"][settings["condition"][n]], n=n)

        for ep in episode_data:
            for stp, ll in zip(ep, [actions, rewards, obs0, obs1, terminated, avail_actions]):
                ll.append(stp)

    return actions, rewards, obs0, obs1, terminated, avail_actions


import scipy

def loglikelihood_binary(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    (
        agent,
        parameter_names,
        agent_settings,
        actions,
        rewards,
        starting,
        obs,
        terminated,
        avail_actions,
    ) = args

    agent_settings.update({i: j for i, j in zip(parameter_names, x)})

    agent = agent(**agent_settings)

    # Initialize values
    logp_actions = np.zeros(len(actions))

    for t, (a, r, o, ot1, term, ava) in enumerate(
        zip(actions, rewards, starting, obs, terminated, avail_actions)
    ):
        # Apply the softmax transformation
        logp_action = np.log(agent.get_probs(o, ava) + np.finfo(float).eps)

        logp_actions[t] = logp_action[a]
        agent.update(o, a, r, term, ot1)

    # Return the negative log likelihood of all observed actions
    return -np.sum(logp_actions[:])


def optimize_loglikelihood(
    actions,
    rewards,
    obs0,
    obs1,
    terminated,
    avail_actions,
    agent_settings,
    parameter_names,
    parameter_settings,
    agent,
    method="L-BFGS-B",
):
    initial_params = [parameter_settings[i]["initial"] for i in parameter_names]
    bounds = tuple(parameter_settings[i]["bounds"] for i in parameter_names)

    result = scipy.optimize.minimize(
        loglikelihood_binary,
        initial_params,
        args=(
            agent,
            parameter_names,
            agent_settings,
            actions,
            rewards,
            obs0,
            obs1,
            terminated,
            avail_actions,
        ),
        method=method,
        bounds=bounds,
    )

    return result