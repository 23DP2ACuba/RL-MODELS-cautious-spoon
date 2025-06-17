# 📄 Q-Learning Stock Trading Agent – Documentation

## 🔧 Technologies Used
- **Python**
- **TensorFlow (Keras)** – Deep Q-network for Q-value approximation
- **NumPy** – Numerical operations and state transformations
- **yfinance** – Stock market data extraction
- **OS** – TensorFlow optimization config

---

## 🚀 Overview

This script implements a **Q-learning agent** that learns how to trade a stock (e.g., GOOGL) by interacting with historical data and maximizing cumulative portfolio value. It uses a **neural network** to approximate the Q-values of actions: `Buy`, `Sell`, or `Hold`.

---

## 🧠 Core Components

### 1. `DecisionPolicy` (Abstract Class)
Defines the interface for trading strategies:
- `select_action(current_state, step)`: Picks an action based on current state.
- `update_q(state, action, reward, next_state)`: Updates Q-values with feedback.

---

### 2. `QLearningDecisionPolicy`
Implements a **deep Q-learning policy** using:
- A 2-layer neural network:
  - Hidden layer: 200 units with ReLU
  - Output layer: One Q-value per action
- **Epsilon-greedy** strategy for exploration/exploitation
- **Bellman update rule** for training:

  \[
  Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')
  \]

#### Parameters:
- `epsilon = 0.5` – Exploration rate
- `gamma = 0.001` – Discount factor
- Optimizer: SGD
- Loss function: MSE

---

### 3. `run_simulation()`

Runs a full trading simulation.

#### Arguments:
- `policy`: The trading strategy (QLearningDecisionPolicy)
- `initial_budget`: Starting cash
- `initial_num_stocks`: Initial stock holdings
- `data`: Stock closing prices
- `hist`: Number of previous days used to form a state

#### Returns:
- `avg`: Average total portfolio value during simulation
- `std`: Standard deviation of portfolio value

---

## 📉 Simulation Flow

1. Initialize budget and stock count.
2. At each timestep:
   - Build current state from `hist` past days.
   - Select an action: `Buy`, `Sell`, or `Hold`.
   - Execute action and receive reward.
   - Update Q-values using the neural network.
   - Track total portfolio value.

---

## 📈 Example Usage

```python
policy = QLearningDecisionPolicy(actions=["Buy", "Sell", "Hold"], input_dim=200)
data = yfinance.Ticker('GOOGL').history(start="2000-01-01", end="2017-01-01")['Close'].values
avg, std = run_simulation(policy, 1000.0, 0, data, hist=200)

print(f"Average Total Value: {avg:.2f}")
print(f"Standard Deviation: {std:.2f}")
