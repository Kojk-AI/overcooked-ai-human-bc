import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import time
import pickle
import overcooked
from src.overcooked_ai_py.mdp.actions import Action

from src.human_aware_rl.rllib.rllib import load_agent, load_policy
from src.human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model
from src.human_aware_rl.imitation.behavior_cloning_tf2 import _get_base_ae, BehaviorCloningPolicy
from src.human_aware_rl.rllib.rllib import RlLibAgent

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

import os


import models

class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def discounted_cumulative_sums(self,x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )

class Trainer():
    def __init__(self):
        self.env = models.Env()
        self.action_list = [Action.INTERACT,Action.STAY,Action.INDEX_TO_ACTION[0],Action.INDEX_TO_ACTION[1],Action.INDEX_TO_ACTION[2],Action.INDEX_TO_ACTION[3]]
        self.overcooked_env = overcooked.Overcooked()
        # Hyperparameters of the PPO algorithm
        self.steps_per_epoch = 400
        self.epochs = 500
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3
        self.train_policy_iterations = 80
        self.train_value_iterations = 80
        self.lam = 0.97
        self.target_kl = 0.01
        self.hidden_sizes = (64, 64)

        # True if you want to render the environment
        self.render = False

        # Initialize the environment and get the dimensionality of the
        # observation space and the number of possible actions
        self.observation_dimensions = 256
        self.num_actions = 6

        self.critic = self.create_critic()
        self.actor = self.create_behaviour_head()
        # self.actor.load_weights("Fixed_bc_pretrained_agent/actor")

        # Initialize the buffer
        self.buffer = Buffer(self.observation_dimensions, self.steps_per_epoch)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

        self.sensor = models.Sensor()



    def create_critic(self):
        # Initialize the actor and the critic as keras models
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = self.mlp(observation_input, list(self.hidden_sizes) + [self.num_actions], tf.tanh, None)
        # actor = keras.Model(inputs=observation_input, outputs=logits)
        value = tf.squeeze(
            self.mlp(observation_input, list(self.hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        return keras.Model(inputs=observation_input, outputs=value)

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self,observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
        self,observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    def mlp(self,x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)

    def create_behaviour_head(self,):
        ## Parse custom network params
        size_hidden_layers = 64
        cell_size = 256
        ## Create graph of custom network. It will under a shared tf scope such that all agents
        ## use the same model
        input_state = tf.keras.Input(
            shape=(cell_size), name="embeddings"
        )
        out = input_state
        # out = tf.keras.layers.Concatenate()([input_last_state,out])
        out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
        out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
        # out = tf.keras.layers.Dense(size_hidden_layers, activation='softmax')(out)
        # out = tf.keras.layers.Dense(size_hidden_layers, activation='softmax')(out)
        out = tf.keras.layers.Dense(6)(out)

        model = tf.keras.Model(input_state, out)

        return model

    def logprobabilities(self,logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    # Sample action from actor
    @tf.function
    def sample_action(self,observation):
        logits = self.actor(observation)
        # action = tf.squeeze(tf.random.categorical(logits, 1), axis=1
        action = tf.argmax(tf.keras.layers.Softmax()(tf.squeeze(logits)))
        return logits, action

    def train_epoch(self, epoch, render):
        # Initialize the observation, episode return and episode length
        # print("Resetting")
        observation, episode_return, episode_length = self.env.reset(), 0, 0
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        replay = []
        # Iterate over the steps of each epoch
        for t in range(self.steps_per_epoch):
            # print("Timestep: {}".format(t))
            # if render:
            #     try:
            #         self.env.render(t)
            #     except Exception as e:
            #         print(e)

            # Get the logits, action, and take one step in the environment
            # observation = observation.reshape(1, -1)
            logits, action = self.sample_action(observation)
            observation_new, reward, done, info = self.env.step(action.numpy())
            reward_sparse, shaped_r = reward
            # rate = np.max((0.0,(1.0-epoch*1e-6)))
            rate = 0
            rewards_sum = reward_sparse + shaped_r*rate
            episode_return += rewards_sum 
            episode_length += 1
            replay.append(info)

            # if render:
            #     try:
            #         self.env.render(t, info)
            #     except:
            #         pass
            # Get the value and log-probability of the action
            value_t = self.critic(observation)
            logprobability_t = self.logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            self.buffer.store(observation, action, rewards_sum, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == self.steps_per_epoch - 1):
                last_value = 0 if done else self.critic(observation.reshape(1, -1))
                self.buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = self.env.reset(), 0, 0

        # # Get values from the buffer
        # (
        #     observation_buffer,
        #     action_buffer,
        #     advantage_buffer,
        #     return_buffer,
        #     logprobability_buffer,
        # ) = self.buffer.get()

        # # Update the policy and implement early stopping using KL divergence
        # for _ in range(self.train_policy_iterations):
        #     kl = self.train_policy(
        #         observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        #     )
        #     if kl > 1.5 * self.target_kl:
        #         # Early Stopping
        #         break

        # # Update the value function
        # for _ in range(self.train_value_iterations):
        #     self.train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print("Rate:{}".format(rate))
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

        return replay

    def run_overcooked(self, actions):
        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False

        replay = []
        # Iterate over the steps of each epoch
        for t in range(self.steps_per_epoch):
            # if self.render:
            #     self.overcooked_env.render()

            # Get the logits, action, and take one step in the environment
            action1 = self.action_list[np.argmax(actions[t][1])]
            action2 = self.action_list[np.argmax(actions[t][2])]
            action_pair = [action1,action2]
            observation_new, reward, done, env_info = self.overcooked_env.step(action_pair)
            shaped_r = sum(env_info['shaped_r_by_agent'])
            episode_return += reward
            episode_length += 1
            replay.append((models._preprocess(self.overcooked_env.mdp.lossless_state_encoding(observation, debug=False)[0]),reward,shaped_r))
            
            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == self.steps_per_epoch - 1):
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0

        # Print mean return and length for each epoch
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

        return replay
    
    def train(self, epochs):
        x_state = []
        x_state_p1 = []
        x_state_p2 = []
        y_state = []
        x_r = []
        x_r_p1 = []
        x_r_p2 = []
        y_r = []
        x_r2 = []
        x_r2_p1 = []
        x_r2_p2 = []
        y_r2 = []
        prev_len1 = 0
        prev_len2 = 0
        prev_len3 = 0

        reward_hist = []
        reward_hist_s = []        

        for i in range(epochs):
            print("")
            print("Epoch: {}".format(i))
            # if self.render and (i+1)%10==0:
            #     replay = self.train_epoch(True)
            # else:
            #     replay = self.train_epoch(False)
            replay = self.train_epoch(epoch=i,render=False)
            # replay_gt = self.run_overcooked(replay)
            # t=0
            # check=False
            # while check == False:
            #     # print("replay_{}".format(t))
            #     # print(len(replay))
            #     # for l in range(34):
            #     #     print(l)
            #     #     print(replay[t][0][...,l])
            #     #     print(replay_gt[t][0][...,l])      
            #     replay[t][0][...,33] = 0      
            #     replay_gt[t][0][...,33] = 0     
            #     if np.array(replay[t][0]==replay_gt[t][0]).all()==False:
            #         check=True
            #         print("Frames before error: {}".format(t))
            #         x_state.append(replay[t-1][0])
            #         x_state_p1.append(replay[t-1][1])
            #         x_state_p2.append(replay[t-1][2])
            #         y_state.append((replay_gt[t][0].astype(int)^replay_gt[t-1][0].astype(int)).astype(float))
            #     if np.array(replay[t][3]==replay_gt[t][1]).all()==False:
            #         x_r.append(replay[t][0])
            #         x_r_p1.append(replay[t][1])
            #         x_r_p2.append(replay[t][2])
            #         print("Sparse")
            #         print(replay_gt[t][1])
            #         if replay_gt[t][1] > 0:
            #             r = 1
            #         else: 
            #             r = 0
            #         y_r.append(r)
            #     if np.array(np.round(replay[t][4])==np.round(replay_gt[t][2])).all()==False:
            #         print("shaped")
            #         print(replay_gt[t][2])
            #         x_r2.append(replay[t][0])
            #         x_r2_p1.append(replay[t][1])
            #         x_r2_p2.append(replay[t][2])
            #         y_r2.append(replay_gt[t][2])
            #     t+=1
            #     if t>=400:
            #         check = True
            
            # if len(x_state)-prev_len1>10:
            #     # print("Num of wrong predicted frame: {}".format(len(x_state)-prev_len1))
            #     # print("Training dynamics")
            #     # self.train_dynamics(x_state, x_state_p1, x_state_p2, y_state)
            #     prev_len1 = len(x_state)
            # if len(x_r)-prev_len2>10:
            # #     # print("Num of wrong rewards: {}".format(len(x_r)-prev_len2))
            #     # print("Training sparse rewards")
            #     # self.train_rewards(x_r, x_r_p1, x_r_p2, y_r)
            #     prev_len2 = len(x_r)
            # if len(x_r2)-prev_len3>10:
            # #     print("Training shaped rewards")
            # # #     # print("Num of wrong rewards: {}".format(len(x_r)-prev_len2))
            # #     self.train_rewards2(x_r2, x_r2_p1, x_r2_p2, y_r2)
            #     prev_len3 = len(x_r2)

            # if len(x_state) > 40000:
            #     x_state = x_state[-40000:]
            #     x_state_p1 = x_state_p1[-40000:]
            #     x_state_p2 = x_state_p2[-40000:]
            #     y_state = y_state[-40000:]
            # if len(x_r) > 40000:
            #     x_r = x_r[-40000:]
            #     x_r_p1 = x_r_p1[-40000:]
            #     x_r_p2 = x_r_p2[-40000:]
            #     y_r = y_r[-40000:]

            # if (i+1)%100==0:
            #     # self.eval_self()
            #     self.actor.save_weights("weights_temp4/actor_{}".format(i))

            r_sparse = 0
            r_shape = 0
            for t in range(400):
                r_sparse += replay[t][3]
                r_shape += replay[t][4]            
            reward_hist.append(r_sparse)
            reward_hist_s.append(r_shape)

            # self.actor.save_weights("weights_temp4/actor")

            # np.save("weights_temp4/reward_hist", np.array(reward_hist))
            # np.save("weights_temp4/reward_hist_s", np.array(reward_hist_s))

            # self.env.enc.save_weights("weights_temp/encoder2")
            # self.env.dec.save_weights("weights_temp/decoder2")
            # self.env.dynamics.save_weights("weights_temp/dynamics2")
            # self.env.rewards.save_weights("weights_temp/rewards2")
            # self.env.rewards_s.save_weights("weights_temp/rewards_s2")

    def train_rewards(self, x_r, x_r_p1, x_r_p2, y_r):
        self.env.rewards.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 5e-4),  # Optimizer
            # Loss function to minimize
            loss=[(
                    tf.keras.losses.BinaryCrossentropy()
            )],
            # List of metrics to monitor
            metrics=['accuracy'],
        )
        self.env.rewards.fit(
            (self.env.enc(np.array(x_r)),np.squeeze(np.array(x_r_p1)),np.squeeze(np.array(x_r_p2))),
            np.array(y_r),
            batch_size=32,
            epochs=5,
        )

    def train_rewards2(self, x_r, x_r_p1, x_r_p2, y_r):
        self.env.rewards_s.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 5e-4),  # Optimizer
            # Loss function to minimize
            loss=[(
                    tf.keras.losses.MeanSquaredError()
            )],
            # List of metrics to monitor
            metrics=['mse'],
        )
        self.env.rewards_s.fit(
            (self.env.enc(np.array(x_r)),np.squeeze(np.array(x_r_p1)),np.squeeze(np.array(x_r_p2))),
            np.array(y_r),
            batch_size=32,
            epochs=5,
        )

    def train_dynamics(self, x_state, x_state_p1, x_state_p2, y_state):
        self.env.enc.trainable=False
        self.env.dec.trainable=False
        vae = VAE(self.env.enc, self.env.dec, self.env.dynamics)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5))
        vae.fit((np.array(x_state),np.squeeze(np.array(x_state_p1)),np.squeeze(np.array(x_state_p2))),np.array(y_state), epochs=20, batch_size=32, shuffle=True)

    def eval_self(self):
        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        self.sensor.reset()
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False
        # replay = []
        # Iterate over the steps of each epoch

        self.actor.load_weights("weights_temp2/actor")

        for t in range(4000):
            # if self.render:
            #     self.overcooked_env.render(t)

            obs = self.overcooked_env.mdp.lossless_state_encoding(observation, debug=False)[0]
            obs = models._preprocess(obs)

            obs2 = obs.copy()

            obs2[...,0] = obs[...,1]
            obs2[...,1] = obs[...,0]
            obs2[...,2] = obs[...,6]
            obs2[...,3] = obs[...,7]
            obs2[...,4] = obs[...,8]
            obs2[...,5] = obs[...,9]
            obs2[...,6] = obs[...,2]
            obs2[...,7] = obs[...,3]
            obs2[...,8] = obs[...,4]
            obs2[...,9] = obs[...,5]

            obs = np.expand_dims(obs,0)
            obs2 = np.expand_dims(obs2,0)

            hidden = self.sensor.sense(obs,0)
            hidden2 = self.sensor.sense(obs2,1)

            action1 = self.actor(hidden)
            action1 = self.action_list[np.argmax(tf.keras.layers.Softmax()(action1[0]).numpy())]

            action2 = self.sensor.action(hidden2)
            action2 = self.action_list[action2]

            action_pair = [action1,action2]
            # print(action_pair)
            observation_new, reward, done, _ = self.overcooked_env.step(action_pair)
            episode_return += reward
            episode_length += 1
            # replay.append(observation,reward)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == self.steps_per_epoch - 1):
                print("Reward for episode: {}".format(episode_return))
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
                self.sensor.reset()

        # Print mean return and length for each epoch
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

    def _agent(self):
        agent_path = "src/overcooked_demo/server/static/assets/agents/RllibCrampedRoomSP/agent"
        # The first argument is the path to the saved trainer, we then loads the agent associated with that trainner
        ## If you use the experiment setup provided, the saved path should be the results_dir in the configuration
        # The second argument is the type of agent to load, which only matters if it is not a self-play agent 
        # The third argument is the agent_index, which is not directly related to the training
        ## It is used in creating the RllibAgent class that is used for evaluation
        ppo_agent = load_agent(agent_path,"ppo",1)
        return ppo_agent

    def _agent2(self):
        agent_path = "src/overcooked_demo/server/static/assets/agents/RllibCrampedRoomSP/agent"
        # The first argument is the path to the saved trainer, we then loads the agent associated with that trainner
        ## If you use the experiment setup provided, the saved path should be the results_dir in the configuration
        # The second argument is the type of agent to load, which only matters if it is not a self-play agent 
        # The third argument is the agent_index, which is not directly related to the training
        ## It is used in creating the RllibAgent class that is used for evaluation
        ppo_agent = load_agent(agent_path,"ppo",0)
        return ppo_agent


    def _bc_agent(self):
        # bc_model_path = "/home/kenneth-stengg/cMBRL/bc_dir_cramped_room"
        bc_model_path = "bc_runs/train/cramped_room"
        bc_model, bc_params = load_bc_model(bc_model_path)

        bc_policy = BehaviorCloningPolicy.from_model(
                bc_model, bc_params, stochastic=True
            )
        # We need the featurization function that is specifically defined for BC agent
        # The easiest way to do it is to create a base environment from the configuration and extract the featurization function
        # The environment is also needed to do evaluation

        # base_ae = _get_base_ae(bc_params)
        # base_env = base_ae.env

        bc_agent = RlLibAgent(bc_policy,1,self.overcooked_env.base.featurize_state_mdp)
        return bc_agent

    # def _bc_agent2(self):
    #     bc_model_path = "src/overcooked_demo/server/static/assets/agents/RllibCrampedRoomBC/agent"
    #     bc_policy = load_policy(bc_model_path,"bc",1)
    #     bc_agent = RlLibAgent(bc_policy,1,self.overcooked_env.featurize_state_mdp)
    #     return bc_agent
    
    def eval_ppo(self, suffix):
        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False
        # replay = []
        # Iterate over the steps of each epoch
        # for suffix in ['99','199','299','399','499','599','699','799','899','999']:
        # with open("weights_temp_ppo_temp_10_run3_eval_bc.txt", "w") as f:
        #     f.close()

        # for suffix in range(500):
        #     if (suffix+1)%10==0 or suffix==0:
        self.actor.load_weights("weights_ppo_forced_80/actor"+ "_" + str(suffix))
        # self.actor.load_weights("weights_temp2/actor")
        ppo = True
        if ppo:
            agent = self._agent()
        else:
            agent = self._bc_agent()
        agent.reset()
        # agent2 = self._agent2()
        # agent2.reset()
        sensor = models.Sensor()
        sensor.reset()

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False

        r_l = []

        traj = []
        replay = []

        for t in range(4000):
            # if self.render:
            #     self.overcooked_env.render(t)

            obs = self.overcooked_env.mdp.lossless_state_encoding(observation, debug=False)[0]
            obs = models._preprocess(obs)

            # obs2 = obs.copy()

            # obs2[...,0] = obs[...,1]
            # obs2[...,1] = obs[...,0]
            # obs2[...,2] = obs[...,6]
            # obs2[...,3] = obs[...,7]
            # obs2[...,4] = obs[...,8]
            # obs2[...,5] = obs[...,9]
            # obs2[...,6] = obs[...,2]
            # obs2[...,7] = obs[...,3]
            # obs2[...,8] = obs[...,4]
            # obs2[...,9] = obs[...,5]

            obs = np.expand_dims(obs,0)
            # obs2 = np.expand_dims(obs2,0)

            hidden = sensor.sense(obs,0)
            # hidden2 = self.sensor.sense(obs2,1)

            action1 = self.actor(hidden)
            action1 = self.action_list[np.argmax(tf.keras.layers.Softmax()(action1[0]).numpy())]
            # action1 = agent2.action(observation)[0]
            # action1 = self.action_list[np.random.randint(0,6)]
            # action2 = self.sensor.action(hidden2)
            # action2 = self.action_list[action2]
            action2 = agent.action(observation)[0]

            action_pair = [action1,action2]
            # print(action_pair)
            observation_new, reward, done, _ = self.overcooked_env.step(action_pair)
            episode_return += reward
            episode_length += 1
            # replay.append(observation,reward)
            info = {}
            info['state'] = observation
            info['action'] = action_pair
            replay.append(info)
            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == self.steps_per_epoch - 1):
                print("Reward for episode: {}".format(episode_return))
                # print(episode_return)
                r_l.append(episode_return)
                traj.append(replay)
                replay = []
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
                self.sensor.reset()

        # Print mean return and length for each epoch
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

        # with open("cramped_room/weights_temp_ppo_10_run2_500_eval_contbc_traj", 'wb') as f:
        #     pickle.dump(traj, f)
        # with open("weights_temp_ppo_temp_10_run3_eval_bc.txt", "a") as f:
        #     f.write(str(sum_return / num_episodes))
        #     f.write('\n')
        return sum_return/num_episodes

def main(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    # trainer.eval_self()
    # trainer.train(1)
    trainer = Trainer()
    resume = False
    resume_step = 0
    if resume:
        resume_step = 425
    else:
        with open("weights_ppo_forced_80.txt", "w") as f:
            f.close()
    for suffix in range(resume_step,500,1):
        if (suffix+1)%10==0 or suffix==0:
            res = trainer.eval_ppo(suffix)
            with open("weights_ppo_forced_80.txt", "a") as f:
                f.write(str(res))
                f.write('\n')

if __name__ == "__main__":
    gpu='-1'
    main(gpu)