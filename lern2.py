import gym
import numpy as np
import datetime
start = datetime.datetime.now()

env = gym.make('FrozenLake-v0')

gamma = 0.9


def value_iteration(env, gamma):
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 10000
    threshold = 1e-20
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        QQ = []
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma *
                                                              updated_value_table[next_state])))
                Q_value.append(np.sum(next_states_rewards))
            value_table[state] = max(Q_value)
            QQ.append(Q_value)
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print('Value-iteration converged at iteration# %d.' %(i+1))
            policy = []
            for state in range(env.observation_space.n):
                policy.append(np.argmax(QQ[state]))
            break
    return policy, QQ, value_table


policy, QQ, v = value_iteration(env, gamma)

end = datetime.datetime.now()
print(end-start)

