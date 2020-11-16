import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_DDPG import DDPG
from time import sleep
from tqdm import tqdm
from simple_quad_MODEL import simple_quad_model as quad_model

I = {'Ix': 8.276e-3, 'Iy': 8.276e-3, 'Iz': 1.612e-2}
h = 0.02
env = quad_model(0.9, 10, I, 0.175, h)

s_dims = 9
a_dims = 4
upper_bound = 6.575
lower_bound = 0

total_episodes = 200
ep = 0

RL = DDPG(upper_bound, lower_bound, s_dims, a_dims)
# RL.load_weight()

ep_reward_list = []
avg_reward_list = []
fig_num = 0
while True:
    prev_state = env.reset()
    episodic_reward = 0
    step = 0
    episode_state = []
    episode_action = []

    for step in tqdm(range(100)):
        # for step in range(100):
        tf_pre_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = RL.action_policy(tf_pre_state)
        # print(action)
        state, reward, done = env.reinforce_step(action)
        episode_state.append(state)
        episode_action.append(action)

        # print(state)

        RL.record(prev_state, action, reward, state)
        episodic_reward += reward

        RL.learn()
        RL.soft_update()

        step += 1

        # if done:
        #     break

        prev_state = state
        sleep(1e-10)

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-step:])
    print("Episode * {} *  Reward is ==> {}".format(ep, episodic_reward))
    # print(step)
    avg_reward_list.append(avg_reward)

    # 画图
    episode_state = np.array(episode_state)
    episode_action = np.array(episode_action)
    plt.subplot(221)
    plt.plot(episode_state[:, 0], 'b')
    plt.plot(episode_state[:, 1], 'g')
    plt.plot(episode_state[:, 2], 'r')
    plt.subplot(222)
    plt.plot(episode_state[:, 3], 'b')
    plt.plot(episode_state[:, 4], 'g')
    plt.plot(episode_state[:, 5], 'r')
    plt.subplot(223)
    plt.plot(episode_state[:, 6], 'b')
    plt.plot(episode_state[:, 7], 'g')
    plt.plot(episode_state[:, 8], 'r')
    plt.subplot(224)
    plt.plot(episode_action[:, 0], 'b')
    plt.plot(episode_action[:, 1], 'g')
    plt.plot(episode_action[:, 2], 'r')
    plt.plot(episode_action[:, 3], 'y')
    fig_num += 1
    filename = "fig/" + str(fig_num) + ".jpg"
    plt.savefig(filename)
    plt.close()
    ep += 1
    if ep%100 == 0 :
        RL.save_model()
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        filename = "reward/"+str(ep)+".jpg"
        plt.savefig(filename)
    # plt.show()

RL.save_model()

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
