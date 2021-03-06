from network_environment_2 import EnvNetwork
from memory import Memory
from DQN_agent import *
from utils_2 import Utils
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import time
import json

np.random.seed(7)

fr = open('cumm_reward_DSV_lr4_up5_be115_B64_u1.txt', 'w')
fa = open('actions_DSV_lr4_up5_be115_B64_u1.txt', 'w')

# starting time
start = time.time()

iterations = 400                   # total number of iterations
TIME_SLOTS = 400                   # number of time-slots to run simulation
NUM_CHANNELS = 2                    # Total number of channels
NUM_USERS = 3                       # Total number of users
ATTEMPT_PROB = 1                    # attempt probability of ALOHA based  models

memory_size = 10000                 # size of experience replay deque
batch_size = 64                     # Num of batches to train at each time_slot
pretrain_length = batch_size        # this is done to fill the deque up to batch size before training
epsilon = 0.1                       # initial exploration rate
epsilon_min = 0.001                 # final exploration rate
epsilon_decay = 0.98                # rate of exponential decay of exploration
gamma = 0.98                        # discount  factor
step_size = 5                       # length of history sequence for each data point in batch

state_size = NUM_CHANNELS + 2
action_size = NUM_CHANNELS + 1      # length of output  (k+1)
learning_rate = 1e-4
beta = 1                            # temperature (changes from 1 -> 20)
epochs = 1
update_freq = 5
dueling = True                     # duel Q learning - enable or disable

"""
Initializing environment and calling the deep Q network agent and utilities methods
and model for DQN proposed in the multiuser spectrum access paper
Dueling could be turned off with dueling=False
"""
env = EnvNetwork(NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB)

utils = Utils(NUM_USERS, NUM_CHANNELS)

dqn_agent = DQNAgent(step_size, state_size, action_size, dueling, learning_rate)
"""
DOUBLE DQN: we are using two networks to predict Q_values and target Q_values separately
for this, the weights of original model are set as weights of the target model for prediction
"""
# DQN1
model = dqn_agent.model
model.summary()
# DQN2
target_model = dqn_agent.target_model

"""
Memory the experience replay buffer(deque) from which each batch will be sampled
and fed to the neural network for training
"""
memory = Memory(max_size=memory_size)

action = env.sample()

"""
OBSERVATION. The format of obs is:
[(ACK1,REWARD1),(ACK2,REWARD2),(ACK3,REWARD3), ...,(ACKn,REWARDn) , (CAP_CHANNEL1,CAP_CHANNEL2,...,CAP_CHANNEL_k)]
this is the residual capacity of the channel - fact that channel is still available or not
"""
obs = env.step(action)
print("obs: ", obs)

"""
this is input buffer which will be used for  predicting next Q-values
inputs as states   
"""
history_input = deque(maxlen=step_size)

"""
State (size 2K +2) is action i of user_i as one hot vector of length action_size +
residual capacity of channels + ACK (0/1) for each user i
"""
state = utils.state_gen(action, obs)


reward = [i[1] for i in obs[:NUM_USERS]]


for ii in range(batch_size * step_size * 5):
    action = env.sample()
    obs = env.step(action)
    next_state = utils.state_gen(action, obs)
    reward = [i[1] for i in obs[:NUM_USERS]]
    memory.add((state, action, reward, next_state))
    state = next_state
    history_input.append(state)

avg_loss = []
avg_val_loss = []
interval = 1  # debug interval
cum_reward_ep = np.zeros(iterations)
cum_collision_ep = np.zeros(iterations)


all_actions = []

for iteration in range(iterations):
    loss_init = 0
    val_loss_init = 0
    # list of total rewards
    total_reward = []
    # cumulative reward
    cum_reward = [0]

    # cumulative collision
    cum_collision = [0]
    cum_collision_temp = [0]
    cum_reward_temp = [0]
    channel_utilization = np.zeros([TIME_SLOTS], dtype=np.float32)
    
    '''Starting history with random states or -1's
    '''
    for time_step in range(TIME_SLOTS):

        print("******************* EPISODE ", str(iteration), "******************************")
        print("******************* TIME STEP ", str(time_step), "******************************")

        # ACTION with exploration and exploitation
        action = np.zeros([NUM_USERS], dtype=np.int32)

        state_vector = np.array(history_input)
        # print("state vec: ", np.shape(state_vector), state_vector)

        # probabilities for all users - same
        prob_ = np.zeros([NUM_USERS, action_size], dtype=np.float64)

        for user_i in range(NUM_USERS):

            state_vector_i = np.reshape(state_vector[:, user_i], [1, step_size, state_size])

            # Predict Q values for all actions using DQN1
            Q_state = model.predict(state_vector_i)
            #print("Q_state for user " + str(user_i) +":", Q_state)

            # ***************************************************
            """
            Draw action according to the following probability distribution
            given in eq. 11 of the paper - we convert the first value into log form to avoid getting inf. 
            """

            prob_temp = logsumexp(beta*Q_state)      # log sum exp trick
         
            prob_temp = beta*Q_state - prob_temp      # log(x/y) = log(x) - log(y)
      
            prob = (1 - epsilon) * np.exp(prob_temp) + epsilon / (NUM_CHANNELS + 1)

            prob_[user_i] = prob
            if not same_policy:
                print("************ DIFFERENT POLICY MODE ************")
                prob[0] /= prob[0].sum()  # for sum of prob to be equal to 1
                print("pmf for user " + str(user_i)+":", prob[0])
                action[user_i] = np.random.choice(action_size, 1, p=prob[0])

        # now take action as predicted from the Q-values and receiving the observation from thr environment
        obs = env.step(action)
        print("action :", action)
        print("observation: ", obs)

        # generate next state from action and observation
        next_state = utils.state_gen(action, obs)

        # reward for all users
        reward = [i[1] for i in obs[:NUM_USERS]]

        # calculate sum of rewards
        sum_rewards = np.sum(reward)
        channel_utilization[time_step] = sum_rewards / NUM_CHANNELS
        cum_reward.append(cum_reward[-1] + sum_rewards)

        # If NUM_CHANNELS = 2 , total possible reward = 2 ,
        # therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r)
        collision = NUM_CHANNELS - sum_rewards

        cum_collision.append(cum_collision[-1] + collision)

        # ========================================================
        """
        reward for cooperative policy
        give sum reward to users who contributed, and rest 0

        for i in range(len(reward)):
            if reward[i] > 0:
                reward[i] = sum_rewards
        total_reward.append(sum_rewards)
        print("total reward: ", total_reward)
        """
        # ========================================================

        # adding new experiences into buffer memory
        memory.add((state, action, reward, next_state))

        state = next_state
        history_input.append(state)
        # print("state: ", state)

        # =================== TRAINING BLOCK STARTS ====================

        # sampling a batch from memory for training
        batch = memory.sample(batch_size, step_size)

        # user states generator
        # rank 4 matrix of shape, [NUM_USERS,batch_size,step_size,state_size]
        states = utils.get_states_user(batch)

        # rank 3 matrix of size [NUM_USERS,batch_size,step_size]
        actions = utils.get_actions_user(batch)

        # size[NUM_USERS, batch_size, step_size]
        rewards = utils.get_rewards_user(batch)

        # size [NUM_USERS,batch_size,step_size, state_size]
        next_states = utils.get_next_states_user(batch)

        # reshaping [NUM_USERS,batch_size] -> [NUM_USERS * batch_size]
        # to feed into neural network

        states = np.reshape(states, [-1, states.shape[2], states.shape[3]])
        actions = np.reshape(actions, [-1, actions.shape[2]])
        rewards = np.reshape(rewards, [-1, rewards.shape[2]])
        next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])

        # Double Q-learning
        # calculating target Q_values (possible actions at t+1)
        Q_predict = model.predict(states)
        Q_next = model.predict(next_states)
        target_Q_next = target_model.predict(next_states)

        Q_targets = Q_predict
        for i in range(states.shape[0]):
            old_Q_state = Q_targets[i, actions[i, -1]]  # for buffer updated in case of PER priority exp replay
            best_action = np.argmax(Q_next[i, :])
            Q_targets[i, actions[i, -1]] = rewards[i, -1] + gamma*target_Q_next[i, best_action]

        # calculating loss (bellman loss)
        # loss between whatever is predicted by Q_pred and the targets Q_targets
        loss = model.fit(states, Q_targets, validation_split=0.3, verbose=0, epochs=epochs)

        # updating weight after 5 iteration
        if iteration % update_freq == 0 and iteration > 1:
            # transfer_weights()
            target_model.set_weights(model.get_weights())

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # plot after every 100 time slots
        if time_step % 50 == 49:
            cum_collision_temp.append(cum_collision[-1])
            cum_reward_temp.append(cum_reward[-1])

            total_reward = []
            cum_reward = [0]
            cum_collision = [0]

    # for higher number of episodes, we can reduce these after certain time slots
    if beta < 20:
        beta *= 1.15
    else:
        beta = 20

    cum_collision_ep[iteration] = np.sum(cum_collision_temp)
    cum_reward_ep[iteration] = np.sum(cum_reward_temp)

    # print("Avg loss: ", loss_init/TIME_SLOTS)
    avg_loss.append(loss_init / TIME_SLOTS)
    avg_val_loss.append(val_loss_init / TIME_SLOTS)
    
json.dump(cum_reward_ep.tolist(), fr)
fr.close()

json.dump(all_actions, fa)
fa.close()

plt.figure(4)
cum_reward_f, = plt.plot(np.arange(iterations), cum_reward_ep, label="Rewards")
#cum_collision_f, = plt.plot(np.arange(iterations), cum_collision_ep, label="Collisions")
#plt.legend(handles=[cum_reward_f, cum_collision_f])
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards')
plt.savefig('rewd3u2c_DSV_lr4_up5_be115_B64_u1.eps')
print("*************************************************")

# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
