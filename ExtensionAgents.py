
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaQPlusAgent:
    """exploration bonus kappa*sqrt(tau) added to simulated rewards during planning updates
    """

    def __init__(self, n_states, n_actions, learning_rate, gamma, kappa=0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.kappa = kappa
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_counts = np.zeros((n_states, n_actions, n_states))
        self.Rsum = np.zeros((n_states, n_actions, n_states))
        self.tau = np.zeros((n_states, n_actions)) #timesteps since (s,a) was last tried in a real interaction
        self.t = 0  # total timestep counter

    def select_action(self, s, epsilon):
        greedy_prob = np.random.rand()
        if greedy_prob <= epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):
        self.t += 1
        self.tau += 1
        self.tau[s, a] = 0

        self.n_counts[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r

        if done:
            self.Q_sa[s, a] += self.learning_rate * (r - self.Q_sa[s, a])
        else:
            self.Q_sa[s, a] += self.learning_rate * (
                    r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a]
            )

        prev_states = np.argwhere(self.n_counts.sum(axis=2) > 0)
        for _ in range(n_planning_updates):
            i = np.random.randint(len(prev_states))
            s_p, a_p = prev_states[i]

            counts = self.n_counts[s_p, a_p]
            s_bar = np.random.choice(self.n_states, p=counts / counts.sum())
            r_bar = self.Rsum[s_p, a_p, s_bar] / self.n_counts[s_p, a_p, s_bar]

            #exploration bonus
            r_bonus = r_bar + self.kappa * np.sqrt(self.tau[s_p, a_p])

            self.Q_sa[s_p, a_p] += self.learning_rate * (r_bonus + self.gamma * np.max(self.Q_sa[s_bar]) - self.Q_sa[s_p, a_p])

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class DynaQPlusActionSelectionAgent:
    """exploration bonus kappa*sqrt(tau) used only during action selection
    """

    def __init__(self, n_states, n_actions, learning_rate, gamma, kappa=0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.kappa = kappa
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_counts = np.zeros((n_states, n_actions, n_states))
        self.Rsum = np.zeros((n_states, n_actions, n_states))
        self.tau = np.zeros((n_states, n_actions))

    def select_action(self, s, epsilon):
        greedy_prob = np.random.rand()
        if greedy_prob <= epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s] + self.kappa * np.sqrt(self.tau[s]))
        return a


    def update(self, s, a, r, done, s_next, n_planning_updates):
        self.tau += 1
        self.tau[s, a] = 0

        self.n_counts[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r

        if done:
            self.Q_sa[s, a] += self.learning_rate * (r - self.Q_sa[s, a])
        else:
            self.Q_sa[s, a] += self.learning_rate * (r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])

        prev_states = np.argwhere(self.n_counts.sum(axis=2) > 0)

        for _ in range(n_planning_updates):
            i = np.random.randint(len(prev_states))
            s_p, a_p = prev_states[i]

            counts = self.n_counts[s_p, a_p]
            s_bar = np.random.choice(self.n_states, p=counts / counts.sum())
            r_bar = self.Rsum[s_p, a_p, s_bar] / self.n_counts[s_p, a_p, s_bar]

            self.Q_sa[s_p, a_p] += self.learning_rate * (
                    r_bar + self.gamma * np.max(self.Q_sa[s_bar]) - self.Q_sa[s_p, a_p]
            )

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                s = s_prime
            returns.append(R_ep)
        return np.mean(returns)
