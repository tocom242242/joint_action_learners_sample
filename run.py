import time
import numpy as np
import matplotlib.pyplot as plt
from jal import JAL
from policy import EpsGreedyQPolicy
from simple_game import SimpleGame
import ipdb

if __name__ == '__main__':
    nb_episode = 100000

    agent1 = JAL(aid=0, alpha=0.1, policy=EpsGreedyQPolicy(), actions=np.arange(2))  # agentの設定
    agent2 = JAL(aid=1, alpha=0.1, policy=EpsGreedyQPolicy(), actions=np.arange(2))  # agentの設定

    game = SimpleGame()
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        _, r1, r2 = game.step(action1, action2)

        agent1.observe(r1, agent2.previous_action_id)
        agent2.observe(r2, agent1.previous_action_id)

    # plt.plot(np.arange(len(agent1.ev_history)),agent1.ev_history)
    plt.plot(np.arange(len(agent1.ev_history)),agent1.ev_history, label="agent1")
    plt.plot(np.arange(len(agent2.ev_history)),agent2.ev_history, label="agent2")
    plt.ylabel("Difference of EV between action 1 and action 2 \n $|EV(a_1)-EV(a_2)|$")
    plt.xlabel("Episode")
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
