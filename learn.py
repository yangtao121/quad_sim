import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_DDPG import DDPG
from simple_quad_MODEL import simple_quad_model as quad_model

I = {'Ix': 8.276e-3, 'Iy': 8.276e-3, 'Iz': 1.612e-2}
env = quad_model(0.9, 10, I, 0.175)

s_dims = 9
a_dims = 4
upper_bound = 20
lower_bound = -20

total_episodes = 1500

RL = DDPG(upper_bound, lower_bound, s_dims, a_dims)

ep_reward_list = []
avg_reward_list = []
fig_num = 0
for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    step = 0
    episode_state = []

    while step < 1500:
        tf_pre_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = RL.action_policy(tf_pre_state)
        # print(action)
        state, reward, done = env.reinforce_step(action)
        episode_state.append(state)

        # print(state)

        RL.record(prev_state, action, reward, state)
        episodic_reward += reward

        RL.learn()
        RL.soft_update()

        step += 1

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-step:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    # print(step)
    avg_reward_list.append(avg_reward)

    # 画图
    episode_state = np.array(episode_state)
    plt.subplot(121)
    plt.plot(episode_state[:, 0], 'b')
    plt.plot(episode_state[:, 1], 'g')
    plt.plot(episode_state[:, 2], 'r')
    plt.subplot(122)
    plt.plot(episode_state[:, 3], 'b')
    plt.plot(episode_state[:, 4], 'g')
    plt.plot(episode_state[:, 5], 'r')
    fig_num += 1
    filename = str(fig_num) + ".jpg"
    plt.savefig(filename)
    plt.close()
    # plt.show()

RL.save_model()

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()