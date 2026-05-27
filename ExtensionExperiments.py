import time
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent
from ExtensionAgents import DynaQPlusAgent, DynaQPlusActionSelectionAgent
from Helper import LearningCurvePlot, smooth


def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    kappa = 0.001                     # exploration bonus factor
    wind_proportion = 0.9             # stochastic environment only

    n_planning_updates_list = [0, 1,3,5]

    intervals = np.arange(0, n_timesteps + 1, eval_interval)
    times = {
        'DynaAgent': {},
        'DynaQPlusAgent': {},
        'DynaQPlusActionSelectionAgent': {}
    }

    best_n = 5
    plot_compare = LearningCurvePlot(title='Dyna variant comparison')

    dyna_curves, _ = run_repetitions(
        'DynaAgent', n_timesteps, n_repetitions, eval_interval,
        learning_rate, gamma, epsilon, best_n, wind_proportion, kappa
    )
    plot_compare.add_curve(intervals, smooth(np.mean(dyna_curves, axis=0), 5),
                           label=f'Dyna (n={best_n})')

    dynaqplus_curves, _ = run_repetitions(
        'DynaQPlusAgent', n_timesteps, n_repetitions, eval_interval,
        learning_rate, gamma, epsilon, best_n, wind_proportion, kappa
    )
    plot_compare.add_curve(intervals, smooth(np.mean(dynaqplus_curves, axis=0), 5),
                           label=f'Dyna-Q+ planning (n={best_n})')

    dynaqplus_action_curves, _ = run_repetitions(
        'DynaQPlusActionSelectionAgent', n_timesteps, n_repetitions, eval_interval,
        learning_rate, gamma, epsilon, best_n, wind_proportion, kappa
    )
    plot_compare.add_curve(intervals, smooth(np.mean(dynaqplus_action_curves, axis=0), 5),
                           label=f'Dyna-Q+ action (n={best_n})')

    plot_compare.save(name='comparison_dyna.png')


def run_repetitions(agent_type, n_timesteps, n_repetitions, eval_interval, learning_rate, gamma, epsilon, n_planning_updates,
                    wind_proportion, kappa):
    n_eval_intervals = n_timesteps // eval_interval + 1
    curves = np.zeros((n_repetitions, n_eval_intervals))
    runtimes = []

    for rep in range(n_repetitions):
        t_start = time.time()

        env = WindyGridworld(wind_proportion=wind_proportion)
        eval_env = WindyGridworld(wind_proportion=wind_proportion)

        if agent_type == "DynaAgent":
            agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        elif agent_type == "DynaQPlusAgent":
            agent = DynaQPlusAgent(env.n_states, env.n_actions,
                                   learning_rate, gamma, kappa=kappa)
        else:
            agent = DynaQPlusActionSelectionAgent(env.n_states, env.n_actions,
                                                  learning_rate, gamma, kappa=kappa)

        s = env.reset()
        eval_idx = 0

        for t in range(n_timesteps):
            # Evaluate at regular intervals
            if t % eval_interval == 0:
                mean_return = agent.evaluate(eval_env)
                curves[rep, eval_idx] = mean_return
                eval_idx += 1

            # Select action, step, update
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)

            if done:
                s = env.reset()
            else:
                s = s_next

        runtimes.append(time.time() - t_start)

    avg_runtime = np.mean(runtimes)
    return curves, avg_runtime


if __name__ == '__main__':
    experiment()