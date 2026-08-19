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

# Step 13 - bandit_parameter_study
def bandit_parameter_study(n_runs, n_steps, seed, settings):
    """Compare bandit strategies and report final average reward per setting."""
    results = {}

    for setting in settings:
        method = setting["method"]
        param = setting["param"]
        nonstationary = setting.get("nonstationary", False)

        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"

        final_rewards = np.zeros(n_runs, dtype=float)

        for i in range(n_runs):
            run_seed = seed + i
            rng = np.random.default_rng(run_seed)

            # Stationary testbed or zero-initialized nonstationary testbed.
            if nonstationary:
                true_values = np.zeros(10, dtype=float)
            else:
                true_values = create_bandit_testbed(10, run_seed)

            q_values = np.zeros(10, dtype=float)
            action_counts = np.zeros(10, dtype=int)
            preferences = np.zeros(10, dtype=float)

            # Running average reward for the gradient-bandit baseline.
            average_reward = 0.0

            total_reward = 0.0

            for step in range(n_steps):
                timestep = step + 1

                # Select an action according to the requested method.
                if method == "epsilon_greedy":
                    action = epsilon_greedy_action(q_values, param, rng)

                elif method == "constant_step":
                    action = epsilon_greedy_action(q_values, 0.1, rng)

                elif method == "optimistic":
                    if step == 0:
                        q_values = optimistic_initialization(10, param)

                    action = epsilon_greedy_action(q_values, 0.0, rng)

                elif method == "ucb":
                    action = ucb_action_select(
                        q_values, action_counts, timestep, param
                    )

                elif method == "gradient":
                    # Stable softmax action selection.
                    shifted = preferences - np.max(preferences)
                    exp_preferences = np.exp(shifted)
                    probabilities = exp_preferences / np.sum(exp_preferences)
                    action = int(rng.choice(10, p=probabilities))

                else:
                    raise ValueError(f"Unknown bandit method: {method}")

                # Obtain stochastic reward from the selected arm.
                reward = pull_arm(true_values, action, rng)

                # Update estimates/preferences.
                if method == "constant_step":
                    q_values = constant_step_size_update(
                        q_values, action, reward, param
                    )
                    action_counts[action] += 1

                elif method == "optimistic":
                    q_values = constant_step_size_update(
                        q_values, action, reward, 0.1
                    )
                    action_counts[action] += 1

                elif method == "ucb":
                    q_values, action_counts = sample_average_update(
                        q_values, action_counts, action, reward
                    )

                elif method == "epsilon_greedy":
                    q_values, action_counts = sample_average_update(
                        q_values, action_counts, action, reward
                    )

                elif method == "gradient":
                    preferences = gradient_bandit_update(
                        preferences,
                        action,
                        reward,
                        average_reward,
                        param,
                    )

                    # Baseline includes the current reward.
                    average_reward += (reward - average_reward) / timestep

                total_reward += reward

                # Apply nonstationary drift after each step.
                if nonstationary:
                    true_values = apply_random_walk_drift(
                        true_values, 0.01, rng
                    )

            # Mean reward at the final step means the episode's
            # final-step reward for this run.
            final_rewards[i] = reward

        results[label] = float(np.mean(final_rewards))

    return results

# Step 14 - build_gridworld_mdp
def build_gridworld_mdp():
    """Build the classic 4x4 Sutton & Barto gridworld MDP."""
    n_states = 16
    n_actions = 4

    # P[s][a] = [(probability, next_state, reward)]
    P = [[[] for _ in range(n_actions)] for _ in range(n_states)]

    # Actions: north, east, south, west
    directions = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    terminal_states = {0, 15}

    for s in range(n_states):
        row = s // 4
        col = s % 4

        for action in range(n_actions):
            # Terminal states self-loop with zero reward.
            if s in terminal_states:
                P[s][action] = [(1.0, s, 0.0)]
                continue

            dr, dc = directions[action]
            next_row = row + dr
            next_col = col + dc

            # Moves outside the grid leave the agent in the same state.
            if not (0 <= next_row < 4 and 0 <= next_col < 4):
                next_state = s
            else:
                next_state = 4 * next_row + next_col

            P[s][action] = [(1.0, next_state, -1.0)]

    return {
        "n_states": n_states,
        "n_actions": n_actions,
        "P": P,
    }

# Step 15 - iterative_policy_evaluation
def iterative_policy_evaluation(policy, mdp, gamma, theta):
    """Compute the state-value function for a fixed deterministic or stochastic policy."""
    n_states = mdp["n_states"]
    n_actions = mdp["n_actions"]
    P = mdp["P"]

    values = np.zeros(n_states, dtype=float)

    # Determine whether the policy is deterministic or stochastic.
    policy = np.asarray(policy)

    if policy.ndim == 1:
        # Deterministic policy: one action per state.
        policy_probs = np.zeros((n_states, n_actions), dtype=float)
        for state in range(n_states):
            policy_probs[state, int(policy[state])] = 1.0
    elif policy.ndim == 2:
        # Stochastic policy: probability of each action in every state.
        policy_probs = policy
    else:
        raise ValueError("policy must have shape (n_states,) or (n_states, n_actions)")

    while True:
        delta = 0.0
        new_values = values.copy()

        for state in range(n_states):
            value = 0.0

            for action in range(n_actions):
                action_probability = policy_probs[state, action]

                if action_probability == 0.0:
                    continue

                for probability, next_state, reward in P[state][action]:
                    value += action_probability * probability * (
                        reward + gamma * values[next_state]
                    )

            new_values[state] = value
            delta = max(delta, abs(value - values[state]))

        values = new_values

        if delta < theta:
            break

    return values

# Step 16 - greedy_policy_improvement
def greedy_policy_improvement(state_values, mdp, gamma):
    """Return a greedy policy with respect to the given state-value function."""
    n_states = mdp["n_states"]
    n_actions = mdp["n_actions"]
    P = mdp["P"]

    policy = np.zeros(n_states, dtype=int)

    for state in range(n_states):
        action_values = np.zeros(n_actions, dtype=float)

        for action in range(n_actions):
            for probability, next_state, reward in P[state][action]:
                action_values[action] += probability * (
                    reward + gamma * state_values[next_state]
                )

        # np.argmax breaks ties by selecting the smallest action index.
        policy[state] = int(np.argmax(action_values))

    return policy

# Step 17 - policy_iteration
def policy_iteration(mdp, gamma, theta):
    """Find an optimal policy using iterative policy evaluation and improvement."""
    n_states = mdp["n_states"]

    # Start with a deterministic policy choosing action 0 everywhere.
    policy = np.zeros(n_states, dtype=int)

    while True:
        # Evaluate the current policy.
        state_values = iterative_policy_evaluation(
            policy, mdp, gamma, theta
        )

        # Improve the policy greedily with respect to the evaluated values.
        improved_policy = greedy_policy_improvement(
            state_values, mdp, gamma
        )

        # Stop when the policy is stable.
        if np.array_equal(improved_policy, policy):
            return state_values, policy

        policy = improved_policy

# Step 18 - value_iteration
def value_iteration(mdp, gamma, theta):
    """Solve an MDP using value iteration and recover an optimal policy."""
    n_states = mdp["n_states"]
    n_actions = mdp["n_actions"]
    P = mdp["P"]

    state_values = np.zeros(n_states, dtype=float)

    while True:
        delta = 0.0
        new_values = state_values.copy()

        for state in range(n_states):
            action_values = np.zeros(n_actions, dtype=float)

            for action in range(n_actions):
                for probability, next_state, reward in P[state][action]:
                    action_values[action] += probability * (
                        reward + gamma * state_values[next_state]
                    )

            # Bellman optimality backup.
            new_values[state] = np.max(action_values)

            delta = max(
                delta,
                abs(new_values[state] - state_values[state])
            )

        state_values = new_values

        if delta < theta:
            break

    # Recover the deterministic greedy policy from the converged values.
    policy = greedy_policy_improvement(state_values, mdp, gamma)

    return state_values, policy

# Step 19 - build_gambler_mdp
def build_gambler_mdp(goal, head_prob):
    """Build the gambler's-problem MDP as a dynamics dictionary.

    Parameters
    ----------
    goal : int
        Capital target (terminal winning state).
    head_prob : float
        Probability that the coin lands heads.

    Returns
    -------
    mdp : dict
        Keys 'n_states', 'n_actions', and 'P' (dynamics table).
    """
    n_states = goal + 1
    n_actions = goal

    # P[s][a] contains the transition list for action a.
    P = [[] for _ in range(n_states)]

    # Terminal states have one self-loop action.
    P[0] = [[(1.0, 0, 0.0)]]
    P[goal] = [[(1.0, goal, 0.0)]]

    # Build transitions for each non-terminal capital state.
    for s in range(1, goal):
        max_stake = min(s, goal - s)

        for stake in range(1, max_stake + 1):
            next_state_heads = s + stake
            next_state_tails = s - stake

            head_reward = (
                1.0 if next_state_heads == goal else 0.0
            )

            P[s].append([
                (float(head_prob), next_state_heads, head_reward),
                (float(1.0 - head_prob), next_state_tails, 0.0),
            ])

    return {
        "n_states": n_states,
        "n_actions": n_actions,
        "P": P,
    }

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

