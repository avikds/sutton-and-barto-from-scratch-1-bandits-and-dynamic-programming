"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    # Use NumPy's legacy RandomState to match the expected seeded sequence.
    rng = np.random.RandomState(seed)

    # Draw k independent values from N(mean, std^2).
    return rng.normal(loc=mean, scale=std, size=k)

# Step 2 - pull_arm
def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    # Return the selected arm's true value plus independent N(0, 1) noise.
    return float(true_values[action] + rng.normal())

# Step 3 - sample_average_update
def sample_average_update(q_values, action_counts, action, reward):
    # Return updated copies so the original arrays are not modified.
    q_new = np.array(q_values, copy=True)
    counts_new = np.array(action_counts, copy=True)

    # Increment the count for the selected action.
    counts_new[action] += 1

    # Incremental sample-average update:
    # Q_new = Q_old + (reward - Q_old) / N
    q_new[action] += (reward - q_new[action]) / counts_new[action]

    return q_new, counts_new

# Step 4 - epsilon_greedy_action
def epsilon_greedy_action(q_values, epsilon, rng):
    # Explore with probability epsilon.
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))

    # Exploit: np.argmax returns the smallest index on ties.
    return int(np.argmax(q_values))

# Step 5 - run_bandit_episode
def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    k = len(true_values)

    # Initialize action-value estimates and action counts.
    q_values = np.zeros(k, dtype=float)
    action_counts = np.zeros(k, dtype=int)

    # Store rewards and selected actions.
    rewards = np.zeros(n_steps, dtype=float)
    actions = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        # Select an action using epsilon-greedy selection.
        action = epsilon_greedy_action(q_values, epsilon, rng)

        # Pull the selected arm to obtain a stochastic reward.
        reward = pull_arm(true_values, action, rng)

        # Update the action-value estimate using the sample-average rule.
        q_values, action_counts = sample_average_update(
            q_values, action_counts, action, reward
        )

        # Store the result.
        actions[step] = action
        rewards[step] = reward

    return rewards, actions

# Step 6 - track_rewards_and_optimal_actions
def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    k = len(true_values)

    # Initialize action-value estimates and action counts.
    q_values = np.zeros(k, dtype=float)
    action_counts = np.zeros(k, dtype=int)

    # The optimal arm is the one with the highest true action value.
    # np.argmax selects the smallest index in case of a tie.
    optimal_action = int(np.argmax(true_values))

    # Allocate output arrays.
    rewards = np.zeros(n_steps, dtype=float)
    optimal_flags = np.zeros(n_steps, dtype=float)

    for step in range(n_steps):
        # Select an action epsilon-greedily.
        action = epsilon_greedy_action(q_values, epsilon, rng)

        # Pull the selected arm.
        reward = pull_arm(true_values, action, rng)

        # Update the estimated action value using sample averages.
        q_values, action_counts = sample_average_update(
            q_values, action_counts, action, reward
        )

        # Store reward and whether the optimal arm was selected.
        rewards[step] = reward
        optimal_flags[step] = float(action == optimal_action)

    return rewards, optimal_flags

# Step 7 - average_bandit_curves (not yet solved)
# TODO: implement

# Step 8 - apply_random_walk_drift (not yet solved)
# TODO: implement

# Step 9 - constant_step_size_update (not yet solved)
# TODO: implement

# Step 10 - optimistic_initialization (not yet solved)
# TODO: implement

# Step 11 - ucb_action_select (not yet solved)
# TODO: implement

# Step 12 - gradient_bandit_update (not yet solved)
# TODO: implement

# Step 13 - bandit_parameter_study (not yet solved)
# TODO: implement

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

