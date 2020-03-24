import gym
import numpy as np
import datetime

start = datetime.datetime.now()

env = gym.make('FrozenLake-v0')
gammaa = 0.9


def compute_value_function(policy, gamma):
    value_table = np.zeros(env.env.nS)
    threshold = 1e-10
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.env.nS):
            action = policy[state]
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in env.env.P[state]
                                      [action]])
        if np.sum((np.fabs(updated_value_table - value_table))) <= threshold:
            break
    return value_table


def extract_policy(value_table, gamma):
    policy = np.zeros(env.observation_space.n)
    QQ = []
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
                policy[state] = np.argmax(Q_table)
        QQ.append(Q_table)
    return policy, QQ


def policy_iteration(env, gamma):
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    for i in range(no_of_iterations):
        new_value_function = compute_value_function(random_policy, gamma)
        new_policy, QQtable = extract_policy(new_value_function, gamma)
        if np.all(random_policy == new_policy):
            print('Policy-Iteration converged at step {}.'.format(i + 1))
            break
        random_policy = new_policy
    return new_policy, new_value_function, QQtable


a, new_v, Q = policy_iteration(env, gammaa)

end = datetime.datetime.now()
print(end - start)

print(a)
