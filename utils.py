"""
Utility function used for the main algorithm
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import sys


class Utils:

    def __init__(self, num_users, num_channels):
        self.NUM_CHANNELS = num_channels
        self.NUM_USERS = num_users

    # creates one hot vector
    def one_hot(self, num, length):
        assert 0 <= num < length, "error"
        vec = np.zeros([length], np.int32)
        vec[num] = 1
        return vec

    # generates next-state from action and observation
    # state contains action of the user i as one hot vector, residual capacity of all channels, and ACK for user i
    def state_gen(self, action, obs):
        input_vec = []
        if action is None:
            print('None')
            sys.exit()
        for user_i in range(action.size):
            input_vec_i = self.one_hot(action[user_i], self.NUM_CHANNELS+1)
            channel_alloc = obs[-1]
            input_vec_i = np.append(input_vec_i, channel_alloc)
            input_vec_i = np.append(input_vec_i, int(obs[user_i][0]))
            input_vec.append(input_vec_i)
        return input_vec

    def get_states(self, batch):
        states = []
        for i in batch:
            states_per_batch = []
            for step_i in i:
                states_per_step = []
                for user_i in step_i[0]:
                    states_per_step.append(user_i)
                states_per_batch.append(states_per_step)
            states.append(states_per_batch)
        return states

    def get_actions(self, batch):
        actions = []
        for each in batch:
            actions_per_batch = []
            for step_i in each:
                actions_per_step = []
                for user_i in step_i[1]:
                    actions_per_step.append(user_i)
                actions_per_batch.append(actions_per_step)
            actions.append(actions_per_batch)
        return actions

    def get_rewards(self, batch):
        rewards = []
        for each in batch:
            rewards_per_batch = []
            for step_i in each:
                rewards_per_step = []
                for user_i in step_i[2]:
                    rewards_per_step.append(user_i)
                rewards_per_batch.append(rewards_per_step)
            rewards.append(rewards_per_batch)
        return rewards


    def get_next_states(self, batch):
        next_states = []
        for each in batch:
            next_states_per_batch = []
            for step_i in each:
                next_states_per_step = []
                for user_i in step_i[3]:
                    next_states_per_step.append(user_i)
                next_states_per_batch.append(next_states_per_step)
            next_states.append(next_states_per_batch)
        return next_states

    # get states for each user from the batch
    def get_states_user(self, batch):
        states = []
        for user in range(self.NUM_USERS):
            states_per_user = []
            for each in batch:
                states_per_batch = []
                for step_i in each:
                    states_per_step = step_i[0][user]
                    states_per_batch.append(states_per_step)
                states_per_user.append(states_per_batch)
            states.append(states_per_user)
        return np.array(states)

    # get actions for each user from the batch
    def get_actions_user(self, batch):
        actions = []
        for user in range(self.NUM_USERS):
            actions_per_user = []
            for each in batch:
                actions_per_batch = []
                for step_i in each:
                    actions_per_step = step_i[1][user]
                    actions_per_batch.append(actions_per_step)
                actions_per_user.append(actions_per_batch)
            actions.append(actions_per_user)
        return np.array(actions)

    # get rewards for each user from the batch
    def get_rewards_user(self, batch):
        rewards = []
        for user in range(self.NUM_USERS):
            rewards_per_user = []
            for each in batch:
                rewards_per_batch = []
                for step_i in each:
                    rewards_per_step = step_i[2][user]
                    rewards_per_batch.append(rewards_per_step)
                rewards_per_user.append(rewards_per_batch)
            rewards.append(rewards_per_user)
        return np.array(rewards)

    # get next states for each user from the batch
    def get_next_states_user(self, batch):
        next_state = []
        for user in range(self.NUM_USERS):
            next_state_per_user = []
            for each in batch:
                next_state_per_batch = []
                for step_i in each:
                    next_state_per_step = step_i[3][user]
                    next_state_per_batch.append(next_state_per_step)
                next_state_per_user.append(next_state_per_batch)
            next_state.append(next_state_per_user)
        return np.array(next_state)
