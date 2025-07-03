# poop_main_dqn_loop.py

import random
import pickle
from collections import deque  # For Replay Buffer
import numpy as np
import pandas as pd
import sys
import datetime
import json
import os

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from game_logic_env import GameEnv
from dqn_model import DQN, get_state_space_size # <-- Import DQN model and helper

# --- DQN Hyperparameters ---
LEARNING_RATE = 0.001  # Learning rate for the optimizer
BATCH_SIZE = 64  # Size of experiences sampled from replay buffer
GAMMA = 0.99  # Discount Factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY_RATE = 0.99995  # Epsilon decay rate per episode
TARGET_UPDATE_FREQ = 100  # How often to update the target network (in episodes)
BUFFER_SIZE = 10000  # Size of the replay buffer

NUM_EPISODES = 20000  # Total episodes for training
MAX_STEPS_PER_EPISODE = 500  # Max steps per episode to prevent infinite loops

# Determine the device to run on (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters dictionary for saving
HYPERPARAMETERS_DQN = {
    "LEARNING_RATE": LEARNING_RATE,
    "BATCH_SIZE": BATCH_SIZE,
    "GAMMA": GAMMA,
    "EPSILON_START": EPSILON_START,
    "EPSILON_END": EPSILON_END,
    "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE,
    "TARGET_UPDATE_FREQ": TARGET_UPDATE_FREQ,
    "BUFFER_SIZE": BUFFER_SIZE,
    "NUM_EPISODES": NUM_EPISODES,
    "MAX_STEPS_PER_EPISODE": MAX_STEPS_PER_EPISODE,
    "DEVICE": str(DEVICE),  # Convert device object to string for JSON
}

# File path settings
RESULTS_FOLDER_DQN = "dqn_results"  # <-- Dedicated folder for DQN results
MODEL_WEIGHTS_FILE_BASE = "dqn_model_weights"
LOG_FILE_BASE_DQN = "dqn_training_log"
HYPERPARAM_FILE_BASE_DQN = "dqn_hyperparam"


# --- Replay Buffer Class ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        # Convert state tuple to list for consistency, if needed
        # Or ensure that states are always converted to tensors outside
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, batch_size)
        # Separate the components into lists
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- Agent (Policy) ---
def choose_action(state, epsilon, model, env):
    """
    Epsilon-greedy action selection.
    Args:
        state (tuple): Current discrete state from the environment.
        epsilon (float): Exploration rate.
        model (DQN): The current DQN model.
        env (GameEnv): The environment instance to get n_actions.
    Returns:
        int: Chosen action index.
    """
    if random.random() < epsilon:
        # Explore: choose a random action
        return random.randint(0, env.n_actions - 1)
    else:
        # Exploit: choose the action with the highest predicted Q-value
        # Convert the state tuple to a torch tensor.
        # Ensure the state values are floats for the neural network.
        state_tensor = (
            torch.tensor(list(state), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        # Disable gradient calculations for inference
        with torch.no_grad():
            q_values = model(state_tensor)

        # Get the action with the maximum Q-value
        return q_values.argmax().item()


# --- Training Function ---
def train_dqn(load_model_weights=True):
    """
    DQN algorithm to train the agent.
    load_model_weights: True to load existing weights, False to start fresh.
    """
    env = GameEnv(render_mode=False, target_fps=0)

    # Initialize DQN and Target DQN models
    state_size = get_state_space_size(env)  # Get state input size from env
    action_size = env.n_actions

    policy_net = DQN(state_size, action_size).to(DEVICE)
    target_net = DQN(state_size, action_size).to(DEVICE)
    target_net.load_state_dict(
        policy_net.state_dict()
    )  # Target network starts with same weights
    target_net.eval()  # Target network is not trained directly

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # Mean Squared Error loss

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # Create results folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_DQN, exist_ok=True)

    # Load model weights (if requested and file exists)
    current_model_path = os.path.join(
        RESULTS_FOLDER_DQN, f"{MODEL_WEIGHTS_FILE_BASE}.pth"
    )
    if load_model_weights:
        try:
            policy_net.load_state_dict(
                torch.load(current_model_path, map_location=DEVICE)
            )
            target_net.load_state_dict(
                policy_net.state_dict()
            )  # Sync target with loaded policy
            print(
                f"기존 모델 가중치 '{current_model_path}' 로드 완료. 학습을 계속합니다."
            )
        except FileNotFoundError:
            print(
                f"'{current_model_path}' 파일을 찾을 수 없습니다. 새로운 모델로 학습을 시작합니다."
            )
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}. 새로운 모델로 학습을 시작합니다.")
    else:
        print("새로운 모델로 처음부터 학습을 시작합니다.")

    epsilon = EPSILON_START

    episode_logs = []  # To store training progress for Excel

    print("DQN 학습 시작...")

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS_PER_EPISODE):
            action = choose_action(state, epsilon, policy_net, env)
            next_state, reward, done, _ = env.step(action)

            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            # If enough experiences are in the buffer, perform a training step
            if len(replay_buffer) > BATCH_SIZE:
                # Sample a batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    BATCH_SIZE
                )

                # Convert to tensors
                # Convert list of tuples to tensor (each tuple element as float)
                states_t = torch.tensor(
                    [list(s) for s in states], dtype=torch.float32
                ).to(DEVICE)
                actions_t = (
                    torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(DEVICE)
                )  # Ensure actions are (batch_size, 1)
                rewards_t = (
                    torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
                )
                next_states_t = torch.tensor(
                    [list(ns) for ns in next_states], dtype=torch.float32
                ).to(DEVICE)
                dones_t = (
                    torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
                )  # True/False to 1/0

                # Calculate Q-values for current states (from policy_net)
                # Use .gather to select Q-value for the action that was taken
                q_values = policy_net(states_t).gather(1, actions_t)

                # Calculate target Q-values
                # max(1)[0] gets the max Q-value for next_state over all actions
                # .detach() is crucial to prevent gradients from flowing into the target network
                next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(-1)

                # If done, target Q-value is just the reward (no future reward)
                target_q_values = rewards_t + GAMMA * next_q_values * (1 - dones_t)

                # Compute loss
                loss = criterion(q_values, target_q_values.detach())

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Optional: Clip gradients to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Epsilon decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)

        # Update target network
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log episode results
        episode_logs.append(
            {
                "Episode": episode + 1,
                "Score": env.score,  # Game score
                "Total_Reward": total_reward,  # Agent's total reward
                "Epsilon": epsilon,
                "Steps": step + 1,
                "Loss": (
                    loss.item() if "loss" in locals() else np.nan
                ),  # Log loss only if calculated
            }
        )

        if (episode + 1) % 100 == 0 or episode == 0:
            print(
                f"에피소드: {episode+1}/{NUM_EPISODES}, 점수: {env.score}, 총 보상: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Loss: {episode_logs[-1]['Loss']:.4f}"
            )

        # Save intermediate model weights (no timestamp, for continued training)
        if (episode + 1) % 1000 == 0:
            torch.save(policy_net.state_dict(), current_model_path)
            print(
                f"모델 가중치 중간 저장 완료 ({current_model_path}, 에피소드 {episode+1})"
            )

    env.close()

    # --- Final Saves ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save final model weights (with timestamp)
    final_model_path = os.path.join(
        RESULTS_FOLDER_DQN, f"{MODEL_WEIGHTS_FILE_BASE}_{timestamp}.pth"
    )
    torch.save(policy_net.state_dict(), final_model_path)
    print("------------------------------------------")
    print(f"최종 모델 가중치 '{final_model_path}' 저장 완료!")
    print(f"총 에피소드: {NUM_EPISODES}")
    print("------------------------------------------")

    # 2. Save hyperparameters to JSON
    hyperparam_file = os.path.join(
        RESULTS_FOLDER_DQN, f"{HYPERPARAM_FILE_BASE_DQN}_{timestamp}.json"
    )
    try:
        with open(hyperparam_file, "w") as f:
            json.dump(HYPERPARAMETERS_DQN, f, indent=4)
        print(f"하이퍼파라미터가 '{hyperparam_file}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"하이퍼파라미터 JSON 저장 중 오류 발생: {e}")

    # 3. Save training logs to Excel
    log_file_with_timestamp = os.path.join(
        RESULTS_FOLDER_DQN, f"{LOG_FILE_BASE_DQN}_{timestamp}.xlsx"
    )
    try:
        df = pd.DataFrame(episode_logs)
        df.to_excel(log_file_with_timestamp, index=False)
        print(f"학습 로그가 '{log_file_with_timestamp}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"학습 로그 Excel 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    # To load existing weights and continue training:
    train_dqn(load_model_weights=True)

    # To start training from scratch with new weights:
    # train_dqn(load_model_weights=False)

    sys.exit()  # Ensure the script exits cleanly
