#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import time
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    
    wind_proportions = [0.9,1.0]
    n_planning_updatess = [0, 1,3,5] 
    
    wind_labels = {0.9: 'Stochastic', 1.0: 'Deterministic'}
    intervals = np.arange(0, n_timesteps+1, eval_interval)
 
    times = {'DynaAgent': {}, 'PrioritizedSweepingAgent': {}}
    q_curves = {}  

    for wind_prop in wind_proportions:
        label = wind_labels[wind_prop]
        plot = LearningCurvePlot(title=f'Dyna : {label}')
        for n in n_planning_updatess:
            curves, avg_time = run_repetitions('DynaAgent', n_timesteps, n_repetitions, eval_interval, 
                                             learning_rate, gamma, epsilon, n, wind_prop)
            times['DynaAgent'][n] = avg_time

            mean_curve = smooth(np.mean(curves, axis=0), 5)
 
            if n == 0:
                curve_label = 'Qlearning'
                q_curves[wind_prop] = curves 
            else:
                curve_label = f'Dyna n={n}'
                
            plot.add_curve(intervals, mean_curve, label=curve_label)

        plot.save(name=f'dyna_{label}.png')
 
    for wind_prop in wind_proportions:
        label = wind_labels[wind_prop]
        plot = LearningCurvePlot(title=f'Prioritized Sweeping : {label} environment')
        
        q_mean = smooth(np.mean(q_curves[wind_prop], axis=0), 5)
        plot.add_curve(intervals, q_mean, label='Qlearning')
 
        for n in n_planning_updatess[1:]:
            curves, avg_time = run_repetitions('PrioritizedSweepingAgent', n_timesteps, n_repetitions, eval_interval, 
                                             learning_rate, gamma, epsilon, n, wind_prop)
            times['PrioritizedSweepingAgent'][n] = avg_time
 
            mean_curve = smooth(np.mean(curves, axis=0), 5)
            plot.add_curve(intervals, mean_curve, label=f'PS n={n}')
 
        plot.save(name=f'ps_{label}.png')

    best_dynan = 5    
    best_psn = 5
 
    for wind_prop in wind_proportions:
        label = wind_labels[wind_prop]
        plot = LearningCurvePlot(title=f'Comparison : {label} environment')

        q_mean = smooth(np.mean(q_curves[wind_prop], axis=0), 5)
        plot.add_curve(intervals, q_mean, label='Qlearning')

        dyna_curves, _ = run_repetitions('DynaAgent', n_timesteps, n_repetitions, eval_interval, 
                                         learning_rate, gamma, epsilon, best_dynan, wind_prop)
        plot.add_curve(intervals, smooth(np.mean(dyna_curves, axis=0), 5), label=f'Dyna (n={best_dynan})')

        ps_curves, _ = run_repetitions('PrioritizedSweepingAgent', n_timesteps, n_repetitions, eval_interval, 
                                       learning_rate, gamma, epsilon, best_psn, wind_prop)
        plot.add_curve(intervals, smooth(np.mean(ps_curves, axis=0), 5), label=f'PS (n={best_psn})')
        plot.save(name=f'comparison_{label}.png')

    print("Algorithm  |  n_planning  |   Time")
    q_t = times['DynaAgent'].get(0)
    print(f"QL    |  0  |  {q_t}")
    for n in [1, 3, 5]:
        dyna_t = times['DynaAgent'].get(n)
        ps_t = times['PrioritizedSweepingAgent'].get(n)
        print(f"Dyna |  {n}  |  {dyna_t}")
        print(f"PS    |  {n}  |  {ps_t}")


def run_repetitions(agent_type, n_timesteps, n_repetitions, eval_interval, learning_rate, gamma, epsilon, n_planning_updates,
                    wind_proportion):

    n_eval_intervals = n_timesteps // eval_interval + 1  
    curves = np.zeros((n_repetitions, n_eval_intervals))
    runtimes = []
 
    for rep in range(n_repetitions):
        t_start = time.time()

        env = WindyGridworld(wind_proportion=wind_proportion)
        eval_env = WindyGridworld(wind_proportion=wind_proportion)

        if agent_type == "DynaAgent":
            agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        else:
            agent = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)

        s = env.reset()
        eval_idx = 0
 
        for t in range(n_timesteps):
            if t % eval_interval == 0:
                mean_return = agent.evaluate(eval_env)
                curves[rep, eval_idx] = mean_return
                eval_idx += 1
 
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            s = env.reset() if done else s_next
 
        runtimes.append(time.time() - t_start)

    avg_runtime = np.mean(runtimes)
    return curves, avg_runtime
 
if __name__ == '__main__':
    experiment()