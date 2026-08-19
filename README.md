# Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Implement core multi-armed bandit algorithms and dynamic-programming methods from Sutton and Barto. Build stationary and nonstationary bandit testbeds, compare epsilon-greedy, optimistic, UCB and gradient strategies, then solve gridworld and gambler MDPs with policy and value iteration.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** create_bandit_testbed
- [x] **2.** pull_arm
- [x] **3.** sample_average_update
- [x] **4.** epsilon_greedy_action
- [x] **5.** run_bandit_episode
- [x] **6.** track_rewards_and_optimal_actions
- [x] **7.** average_bandit_curves
- [x] **8.** apply_random_walk_drift
- [x] **9.** constant_step_size_update
- [x] **10.** optimistic_initialization
- [x] **11.** ucb_action_select
- [x] **12.** gradient_bandit_update
- [x] **13.** bandit_parameter_study
- [x] **14.** build_gridworld_mdp
- [x] **15.** iterative_policy_evaluation
- [x] **16.** greedy_policy_improvement
- [x] **17.** policy_iteration
- [x] **18.** value_iteration
- [x] **19.** build_gambler_mdp
- [x] **20.** gambler_value_iteration
- [x] **21.** extract_optimal_stakes

## Results

```
True action values: [ 1.764  0.4    0.979  2.241  1.868 -0.977  0.95  -0.151 -0.103  0.411]
Episode mean reward (eps=0.1): 1.0506
Avg reward @200: 1.1731
Optimal action % @200: 0.52
Mean |drift|: 0.00844
Optimistic Q init: [5. 5. 5.] ...
UCB first action: 0
Gradient prefs sample: [ 0.045 -0.005 -0.005]
Parameter study results: {'epsilon_greedy(0.1)': 1.1666507230150729, 'optimistic(5.0)': 1.2177748821441838, 'ucb(2.0)': 1.4999494200552939, 'gradient(0.1)': 1.3928004528026952}
Policy iteration V[0]: 0.0
Policy iteration policy (flat): [0 3 3 2 0] ...
Value iteration V[0]: 0.0
Gambler V at [1, 25, 50, 75, 99] : [0.0021, 0.16, 0.4, 0.64, 0.9643]
Optimal stakes at [1, 25, 50, 75, 99] : [1, 25, 50, 25, 1]
```
