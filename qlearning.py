import json
import random

# --- Helpers for state/quality bucketization ---
def bucketize_quality(value: float) -> str:
    if value < 0.4:
        return "low"
    elif value < 0.7:
        return "medium"
    return "high"

def sentiment_sign(label: str) -> int:
    return 1 if label.upper() == "POSITIVE" else -1

def target_from_sentiment(label: str) -> float:
    return 1.0 if label.upper() == "POSITIVE" else 0.0

def reward_fn(old_quality, new_quality, target):
    # Reward is positive if moving toward the target
    if target > 0.5:
        return (new_quality - old_quality)
    else:
        # for negative target (e.g. NEGATIVE) reward is positive when quality decreases
        return (old_quality - new_quality)

# --- Q-learning agent ---
class QLearningAgent:
    def __init__(self, actions=None, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.q = {}
        self.actions = actions if actions else ["increase", "decrease", "hold"]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def policy(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_vals)
        # if multiple max, pick randomly among them
        max_actions = [a for a, v in zip(self.actions, q_vals) if v == max_q]
        return random.choice(max_actions)

    def act_adjust(self, quality, action):
        if action == "increase":
            return min(1.0, quality + 0.1)
        elif action == "decrease":
            return max(0.0, quality - 0.1)
        return quality

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        future_q = max([self.get_q(next_state, a) for a in self.actions], default=0.0)
        self.q[(state, action)] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def dump(self, path):
        # make keys serializable
        serializable_q = {f"{k[0]}|{k[1]}": v for k, v in self.q.items()}
        with open(path, "w") as f:
            json.dump(serializable_q, f, indent=2)