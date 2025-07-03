# dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    DQN model for the poop avoidance game.
    Receives the state and outputs the Q value of each action
    """

    def __init__(self, state_space_size, action_space_size):
        super(DQN, self).__init__()
        # defining neural network layers
        # input size varies depending on state size
        # first of all, it is simply composed of 3 layers
        self.fc1 = nn.Linear(state_space_size, 128)  # first layer
        self.fc2 = nn.Linear(128, 128)  # second layer
        self.fc3 = nn.Linear(
            128, action_space_size
        )  # uutput layer: Q-values for each action

    def forward(self, state):
        """
        forward pass function - takes a state and computes Q values
        state: game state tensor (batch_size, state_space_size)
        Returns: Q values forr each action (batch_size, action_space_size)
        """
        x = F.relu(self.fc1(state))  # first + active
        x = F.relu(self.fc2(x))  #second + active
        return self.fc3(x)    # last +  active


# function to get state space size from game environment
def get_state_space_size(env_instance):
    """
    Calculates the total size of the flattened discrete state space
    based on the GameEnv's binning.
    This assumes a tuple state where each element is an index within its bin.
    """
    # Example state: (player_x_discrete, poop_x_rel_discrete, poop_y_discrete,
    #                item_x_rel_discrete, item_kind, life_val, quiz_active_val,
    #                speed_effect_discrete)

    # max possible values for each dimension (from game_logic_env)
    player_x_bins = env_instance.player_x_bins
    poop_x_rel_bins = env_instance.poop_x_rel_bins
    poop_y_bins = env_instance.poop_y_bins
    item_x_rel_bins = env_instance.item_x_rel_bins

    # fitem types: 0(none), 1(shield), 2(heart) = 3 categories
    item_kind_bins = 3

    # life values: 1, 2, 3 = 3 categories (max life usually 3)
    life_val_bins = env_instance.player.life  

    # quiz active: 0(inactive), 1(active) = 2 categories
    quiz_active_val_bins = 2

    # speed effect: 0, 1, 2 = 3 categories
    speed_effect_discrete_bins = 3
    # get actual state from environment to check
    dummy_env = env_instance  
    dummy_state = dummy_env._get_discrete_state()

    # status tuple length
    return len(dummy_state)
