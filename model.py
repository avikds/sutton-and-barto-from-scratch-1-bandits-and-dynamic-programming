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

# Step 7 - average_bandit_curves
def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    # Store the results from each independent run.
    all_rewards = np.zeros((n_runs, n_steps), dtype=float)
    all_optimal_flags = np.zeros((n_runs, n_steps), dtype=float)

    for i in range(n_runs):
        run_seed = seed + i

        # Build a fresh testbed and RNG for this run.
        true_values = create_bandit_testbed(k, run_seed)
        rng = np.random.default_rng(run_seed)

        # Run the episode and track rewards/optimal-arm choices.
        rewards, optimal_flags = track_rewards_and_optimal_actions(
            true_values, n_steps, epsilon, rng
        )

        all_rewards[i] = rewards
        all_optimal_flags[i] = optimal_flags

    # Average across independent runs for each time step.
    mean_rewards = np.mean(all_rewards, axis=0)
    mean_optimal_flags = np.mean(all_optimal_flags, axis=0)

    return mean_rewards, mean_optimal_flags

# Step 8 - apply_random_walk_drift
def apply_random_walk_drift(true_values, drift_std, rng):
    # Draw an independent zero-mean Gaussian increment for each arm.
    drift = rng.normal(loc=0.0, scale=drift_std, size=true_values.shape)

    # Apply the random-walk increment elementwise.
    return true_values + drift

# Step 9 - constant_step_size_update
def constant_step_size_update(q_values, action, reward, alpha):
    # Work on a copy so the input array is not modified in place.
    updated_q_values = np.array(q_values, copy=True)

    # Constant step-size update:
    # Q <- Q + alpha * (reward - Q)
    updated_q_values[action] += alpha * (
        reward - updated_q_values[action]
    )

    return updated_q_values

# Step 10 - optimistic_initialization
def optimistic_initialization(k, initial_value):
    # Create an array of k action-value estimates initialized optimistically.
    return np.full(k, initial_value, dtype=float)

# Step 11 - ucb_action_select
def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    # Any unvisited arm is treated as having an infinite UCB score.
    unvisited = action_counts == 0

    if np.any(unvisited):
        # np.argmax returns the smallest index in case of ties.
        return int(np.argmax(unvisited))

    # Calculate the UCB score for every visited arm.
    ucb_scores = q_values + c * np.sqrt(
        np.log(timestep) / action_counts
    )

    # Select the arm with the highest UCB score.
    # np.argmax breaks ties by choosing the smallest index.
    return int(np.argmax(ucb_scores))

# Step 12 - gradient_bandit_update
def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    # Compute a numerically stable softmax policy.
    shifted_preferences = preferences - np.max(preferences)
    exp_preferences = np.exp(shifted_preferences)
    probabilities = exp_preferences / np.sum(exp_preferences)

    # Advantage signal: reward relative to the average reward baseline.
    advantage = reward - average_reward

    # Gradient-bandit preference update.
    updated_preferences = np.array(preferences, copy=True)
    updated_preferences -= alpha * advantage * probabilities
    updated_preferences[action] += alpha * advantage

    return updated_preferences

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

