#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:39:43 2020

@author: majadoon
"""

"""
Environment setup for the multiple access problem. It returns the observations of the users
"""
import numpy as np
import random


class EnvNetwork:

    def __init__(self, num_users, num_channels, attempt_prob):

        self.NUM_CHANNELS = num_channels
        self.NUM_USERS = num_users
        self.ATTEMPT_PROB = attempt_prob
        self.REWARD = 1

        self.action_space = np.arange(self.NUM_CHANNELS + 1)
        self.users_action = np.zeros([self.NUM_USERS], np.int32)
        self.users_observ = np.zeros([self.NUM_USERS], np.int32)
        """
        The world's simplest agent!

        See: https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
        """

    def random_act(self, action_space):
        self.action_space = action_space
        return self.action_space.sample()

    def reset(self):
        pass

    def sample(self):
        x = np.random.choice(self.action_space, size=self.NUM_USERS)
        return x

    def step(self, action):
        assert action.size == self.NUM_USERS, "action and users should have same dimension {}" .format(action)
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1], np.int32)     #0 for no channel access
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0

        for each in action:
            prob = random.uniform(0, 1)
            if prob <= self.ATTEMPT_PROB:
                self.users_action[j] = each   #action
                channel_alloc_frequency[each] += 1
            j += 1
        for i in range(1, len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0
        for i in range(len(action)):
            self.users_observ[i] = channel_alloc_frequency[self.users_action[i]]

            if self.users_action[i] == 0:
                self.users_observ[i] = 0
            if self.users_observ[i] == 1:
                reward[i] = 1
            obs.append((self.users_observ[i], reward[i]))
        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1 - residual_channel_capacity
        #obs.append(residual_channel_capacity)
        return obs
