import numpy as np
import copy
import ipdb
import random

class JAL():
    def __init__(self, 
                 aid=None, 
                 alpha=0.1, 
                 policy=None, 
                 gamma=0.99, 
                 actions=None, 
                 alpha_decay_rate=None, 
                 epsilon_decay_rate=None):

        self.aid = aid
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.previous_action_id = None
        self.q_values = self._init_q_values()
        self.ev = self._init_ev_values()
        self.opponent_action_history = self._init_opponent_action_history()
        self.pi_o = self._init_pi_o()
        self.nb_step = 0
        self.ev_history = []

    def _init_pi_o(self):
        pi_o = {}
        for action in self.actions:
            pi_o[action] = 0

        return pi_o

    def _init_opponent_action_history(self):
        opponent_action_history = {}
        for action in self.actions:
            opponent_action_history[action] = 0

        return opponent_action_history

    def _init_ev_values(self):
        ev = np.repeat(0.0, len(self.actions))

        return ev

    def _init_q_values(self):
        q_values = {}

        return q_values

    def act(self, training=True):
        self.nb_step += 1
        if training:
            action_id = self.policy.select_action(self.ev)
            self.previous_action_id = action_id
            action = self.actions[action_id]
        else:
            action_id = self.policy.select_greedy_action(self.ev)
            action = self.actions[action_id]

        return action

    def get_previous_action(self):
        action = self.actions[self.previous_action_id]

        return action

    def observe(self, reward, opponent_action, is_learn=True):
        if is_learn:
            self.learn(reward, opponent_action)
            self.update_oponent_model(opponent_action)

    def learn(self,reward, opponent_action):
        joint_actions = (self.previous_action_id, opponent_action)
        self.reward_history.append(reward)
        self.q_values[joint_actions] = self.compute_q_value(reward, joint_actions)
        self.ev[self.previous_action_id] = self.compute_ev()

    def update_oponent_model(self, opponent_action):
        self.opponent_action_history[opponent_action] += 1
        for action in self.actions:
            self.pi_o[action] = self.opponent_action_history[action]/self.nb_step

    def compute_ev(self):
        ev = 0 
        self.ev_history.append(abs(self.ev[0]-self.ev[1]))
        for opponent_action in self.actions:
            joint_actions = (self.previous_action_id, opponent_action)
            if joint_actions not in self.q_values.keys():
                self.q_values[joint_actions] = 0.0
            ev += self.pi_o[opponent_action] * self.q_values[joint_actions]

        return ev 

    def compute_q_value(self, reward, joint_actions):
        if joint_actions not in self.q_values.keys():
            self.q_values[joint_actions] = 0.0
        q = self.q_values[joint_actions]
        updated_q = q + (self.alpha * (reward - q))

        return updated_q
