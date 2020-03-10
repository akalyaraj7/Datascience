import gym
import numpy as np

# load the environment
env = gym.make('FrozenLake-v0')

# implementing Q-table learning algorithm
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set the learning parameters
# leanring rate
lr = 0.8
# gamma value
y = 0.95
num_episodes = 2000

# List for storing rewards
rList = []
for i in range(num_episodes):
    # Reset the environment and get the first new observation
    s = env.reset()
    # print(f"Value of s ; {s}")
    # print(f"Partitioning Q tables: {Q[s, :]}")
    rAll = 0
    d = False
    j = 0
    # The Q table learning algorithm
    while j < 99:
        print(f"Value of j: {j}")
        j += 1
        print(f"Randomly generating numbers : ", np.random.randn(1, env.action_space.n) * (1./(i+1)))
        print(f"Partitioning Q tables: {Q[s, :]}")
        print(f"Total part {Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i+1))}")

        # choose an action by greedily choosing an action from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
        print(f"Action choosen : {a}")
        print (f" The reward for taking a specific action : {env.step(a)}")
        # Get new state and reward from environment
        sl, r, d, _ = env.step(a)
        print (f"State,Action before updation {Q[s, a]}")
        # update Q table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[sl, :])  - Q[s,a])
        print(f"State,Action After updation {Q[s, a]} {Q[s, a]}")
        rAll += r
        s = sl
        if d == True:
            break;
    rList.append(rAll)

# print(f"Score over time ; {str(sum(rList)/num_episodes)}")
# print(f"Final Q table values: {Q}")


