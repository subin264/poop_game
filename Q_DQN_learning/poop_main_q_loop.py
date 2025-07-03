import random
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import sys
import datetime
import json
import os  

from game_logic_env import GameEnv
# Q-learning hyperparameter setting (same as before)
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.99995

NUM_EPISODES = 20000
MAX_STEPS_PER_EPISODE = 500

# bundle into hyperparameter dictionary (for JSON storage)
HYPERPARAMETERS = {
    "ALPHA": ALPHA,
    "GAMMA": GAMMA,
    "EPSILON_START": EPSILON_START,
    "EPSILON_END": EPSILON_END,
    "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE,
    "NUM_EPISODES": NUM_EPISODES,
    "MAX_STEPS_PER_EPISODE": MAX_STEPS_PER_EPISODE,
}

# folder name to save the result file
RESULTS_FOLDER = "q_learning_results"  

# file path name
Q_TABLE_FILE_BASE = "q_table_poop_avoidance"
LOG_FILE_BASE = "q_learning_training_log"
HYPERPARAM_FILE_BASE = "q_learning_hyperparam"


# global q_table variable declare
q_table = None


def choose_action(state, epsilon, env):
    """
    Choose actions according to epsilon-greedy policy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, env.n_actions - 1)
    else:
        return np.argmax(q_table[state])


def train_q_learning(load_q_table=True):
    """
    Q- the main function that trains the agent using the Q-learning algorithm.
    load_q_table: If True, load the existing Q-table and continue training.
            if false, training starts from scratch with a new Q-table.
    """
    env = GameEnv(render_mode=False, target_fps=0)

    global q_table
    q_table = defaultdict(lambda: np.zeros(env.n_actions))

    # f result folder not exist, create it. (this is first file save work so check here)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)  

    # Q-table load (load_q_table이 True일 tery )
    # when loading, use basic file name inside 'q_learning_results' folder
    current_q_table_file = os.path.join(
        RESULTS_FOLDER, f"{Q_TABLE_FILE_BASE}.pkl"
    )  # <-- modify path
    if load_q_table:
        try:
            with open(current_q_table_file, "rb") as f:
                loaded_q_table = pickle.load(f)
                for k, v in loaded_q_table.items():
                    q_table[k] = v
            print(
                f" existing Q-table '{current_q_table_file}' loading complete. Continue learning."
            )
        except FileNotFoundError:
            print(
                f"'{current_q_table_file}'file not found. Starting a new Q-table"
            )
        except Exception as e:
            print(f" Error loading Q-table: {e}. Starting a new Q-table.")
    else:
        print("start learning from scratch with a new Q-table.")

    epsilon = EPSILON_START

    # initialize the list to store learning logs
    episode_logs = []

    print(" Start learning Q-Learning...")

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS_PER_EPISODE):
            action = choose_action(state, epsilon, env)
            next_state, reward, done, _ = env.step(action)

            old_value = q_table[state][action]
            next_max_q = np.max(q_table[next_state]) if not done else 0
            new_q_value = old_value + ALPHA * (reward + GAMMA * next_max_q - old_value)
            q_table[state][action] = new_q_value

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)

        # save episode log
        episode_logs.append(
            {
                "Episode": episode + 1,
                "Score": env.score,
                "Total_Reward": total_reward,
                "Epsilon": epsilon,
                "Steps": step + 1,
            }
        )

        if (episode + 1) % 100 == 0 or episode == 0:
            print(
                f"에피소드: {episode+1}/{NUM_EPISODES}, 점수: {env.score}, 총 보상: {total_reward:.2f}, Epsilon: {epsilon:.4f}"
            )

        # save intermediate Q-table (save with default file name, specify folder)
        if (episode + 1) % 1000 == 0:
            intermediate_q_table_path = os.path.join(
                RESULTS_FOLDER, f"{Q_TABLE_FILE_BASE}.pkl"
            )  
            with open(intermediate_q_table_path, "wb") as f:
                pickle.dump(dict(q_table), f)
            print(
                f"Q-Table intermediate save complete({intermediate_q_table_path}, episode {episode+1})"
            )

    env.close()

    # Generate a timestamp when learning is complete
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 최종 Q-테이블 저장 (시간 스탬프 포함, 폴더 지정)
    final_q_table_file = os.path.join(
        RESULTS_FOLDER, f"{Q_TABLE_FILE_BASE}_{timestamp}.pkl"
    ) 
    with open(final_q_table_file, "wb") as f:
        pickle.dump(dict(q_table), f)
    print("------------------------------------------")
    print(f"Final Q-table'{final_q_table_file}' save! ")
    print(f" total episode: {NUM_EPISODES}")
    print("------------------------------------------")

    # 2. Save hyperparameter JSON file (with timestamp, specify folder)
    hyperparam_file = os.path.join(
        RESULTS_FOLDER, f"{HYPERPARAM_FILE_BASE}_{timestamp}.json"
    )  
    try:
        with open(hyperparam_file, "w") as f:
            json.dump(HYPERPARAMETERS, f, indent=4)
        print(f"hyprparm '{hyperparam_file}'is sussufull save!")
    except Exception as e:
        print(f"hyperparameter JSON save error occur: {e}")

    # 3. Save learning log as Excel file (with time stamp, specify folder)
    log_file_with_timestamp = os.path.join(
        RESULTS_FOLDER, f"{LOG_FILE_BASE}_{timestamp}.xlsx"
    )  
    try:
        df = pd.DataFrame(episode_logs)
        df.to_excel(log_file_with_timestamp, index=False)
        print(f"learing log '{log_file_with_timestamp}'is sussfull save!.")
    except Exception as e:
        print(f"learing Excel save is eree: {e}")


if __name__ == "__main__":
    train_q_learning(load_q_table=True)
    sys.exit()
