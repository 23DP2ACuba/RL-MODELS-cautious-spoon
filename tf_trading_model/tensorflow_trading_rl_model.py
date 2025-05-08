import yfinance
import numpy as np
import tensorflow as tf
import random
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
class DecisionPolicy:
    def select_action(self, current_state, step):
        pass
    
    def update_q(self, state, action, reward, next_state):
        pass

class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions
    
    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim):
        self.epsilon = 0.5 
        self.gamma = 0.001  
        self.actions = actions  
        output_dim = len(actions) 
        h1_dim = 200 

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(h1_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(output_dim)  
        ])

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')

    def select_action(self, current_state, step):
        """Select an action using epsilon-greedy policy."""
        current_state = current_state.reshape(1, -1) if current_state.ndim == 1 else current_state
        
        threshold = min(self.epsilon, step / 1000.0)
        if random.random() < threshold:
            action_q_vals = self.model.predict(current_state, verbose=0)
            action_idx = np.argmax(action_q_vals[0])
            action = self.actions[action_idx]
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action
    
    def update_q(self, state, action, reward, next_state):
        """Update Q-values using the Q-learning update rule."""
        state = state.reshape(1, -1) if state.ndim == 1 else state
        next_state = next_state.reshape(1, -1) if next_state.ndim == 1 else next_state

        current_q = self.model.predict(state, verbose=0)
        next_q = self.model.predict(next_state, verbose=0)

        target_q = current_q.copy()
        action_idx = self.actions.index(action)
        target_q[0, action_idx] = reward + self.gamma * np.max(next_q)

        self.model.train_on_batch(state, target_q)

def run_simulation(policy, initial_budget, initial_num_stocks, data, hist):
    """
    Run a trading simulation over the given data.

    Parameters:
    - policy: DecisionPolicy instance (e.g., QLearningDecisionPolicy)
    - initial_budget: Starting cash amount
    - initial_num_stocks: Starting number of stocks owned
    - data: Array of stock prices (e.g., closing prices)
    - hist: Number of past days to consider in the state

    Returns:
    - avg: Average total value over the simulation
    - std: Standard deviation of total value
    """
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0
    transitions = [] 
    for i in range(hist, len(data)):
        current_state = data[i - hist:i]  
        current_price = data[i]

        action = policy.select_action(current_state, i)

        if action == "Buy" and budget >= current_price:
            num_stocks += 1
            budget -= current_price
            reward = 0  
        elif action == "Sell" and num_stocks > 0:
            num_stocks -= 1
            budget += current_price
            reward = current_price - data[i - 1]  
        else:
            reward = 0  

        next_state = data[i - hist + 1:i + 1]  

        policy.update_q(current_state, action, reward, next_state)

        share_value = num_stocks * current_price
        total_value = budget + share_value
        transitions.append(total_value)

    avg = np.mean(transitions)
    std = np.std(transitions)
    return avg, std

if __name__ == "__main__":
    data = yfinance.Ticker('GOOGL').history(start="2000-01-01", end="2017-01-01")['Close'].values
    
    actions = ["Buy", "Sell", "Hold"]
    hist = 200  
    initial_budget = 1000.0
    initial_num_stocks = 0

    policy = QLearningDecisionPolicy(actions, input_dim=hist)

    avg, std = run_simulation(policy, initial_budget, initial_num_stocks, data, hist)

    print(f"Average Total Value: {avg:.2f}")
    print(f"Standard Deviation: {std:.2f}")