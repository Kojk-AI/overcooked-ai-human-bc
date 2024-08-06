import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import time
import pickle
import random

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

import overcooked
import os
import models
import env

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, ObjectState, SoupState
from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

NUM_HIDDEN_LAYERS = 3
SIZE_HIDDEN_LAYERS = 64
NUM_FILTERS = 25
NUM_CONV_LAYERS = 3
CELL_SIZE = 256

SCENARIOS = {
    "CrampedRoom":{
        "Agent": "RllibCrampedRoomSP",
        "Layout": "cramped_room"
    },
    "AsymmetricAdvantages":{
        "Agent": "RllibAsymmetricAdvantagesSP",
        "Layout": "asymmetric_advantages"
    },
    "CoordinationRing":{
        "Agent": "RllibCoordinationRingSP",
        "Layout": "coordination_ring"
    },
    "CounterCircuit":{
        "Agent": "RllibCounterCircuit1OrderSP",
        "Layout": "counter_circuit_o_1order"
    },
    "ForcedCoordination":{
        "Agent": "RllibForcedCoordinationSP",
        "Layout": "forced_coordination"
    },
}

class State:
    def __init__(self, players, objects=None):
        self.players = players
        self.objects = objects

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
    def __init__(self, scenario,behaviour="behaviour_best", temperature=1.0):
        self.env = env.Env(scenario,behaviour,temperature)
        self.overcooked_env = overcooked.Overcooked(scenario)
        self.scenario = scenario

        self.action_list = [Action.INTERACT,Action.STAY,Action.INDEX_TO_ACTION[0],Action.INDEX_TO_ACTION[1],Action.INDEX_TO_ACTION[2],Action.INDEX_TO_ACTION[3]]
        # Hyperparameters of the PPO algorithm
        self.steps_per_epoch = 400
        self.epochs = 500
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_learning_rate = 3e-4
        # self.policy_learning_rate = 3e-5
        self.value_function_learning_rate = 1e-3
        self.train_policy_iterations = 80
        self.train_value_iterations = 80
        self.lam = 0.97
        self.target_kl = 0.01
        self.hidden_sizes = (64, 64)

        # True if you want to render the environment
        # self.render = False

        # Initialize the environment and get the dimensionality of the
        # observation space and the number of possible actions
        self.observation_dimensions = CELL_SIZE
        self.num_actions = 6

        self.critic = self.create_critic()
        self.actor = models._create_behaviour_model()
        # self.actor.load_weights("data/{}/behaviour/{}".format(scenario, behaviour))
        # Initialize the buffer
        self.buffer = Buffer(self.observation_dimensions, self.steps_per_epoch)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

        self.sensor = env.Sensor(scenario,behaviour,temperature)

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
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    def _parse_obs(self,obs):
        objects = {}
        #process players
        for i in range(4):
            if np.max(obs[...,i+2]) == 1:
                p1_dir = i
            if np.max(obs[...,i+6]) == 1:
                p2_dir = i  

        try:
            p1_dir = Direction.INDEX_TO_DIRECTION[p1_dir]
        except:
            p1_dir = Direction.NORTH
        try:
            p2_dir = Direction.INDEX_TO_DIRECTION[p2_dir]
        except:
            p2_dir = Direction.NORTH

        p1_loc = (np.nonzero(obs[...,0])[0][0],np.nonzero(obs[...,0])[1][0])
        p2_loc = (np.nonzero(obs[...,1])[0][0],np.nonzero(obs[...,1])[1][0])

        p1_object = None
        p2_object = None

        #process soup
        soup_cook_time = 20
        pot_loc = (2,0)  
        # onions_pot= np.max((np.max(obs[...,16]), np.max(obs[...,18])))
        if len(np.where(obs[...,16:20]>0)[-1]) > 0:
            onion1 = np.where(obs[...,16:20]>0)[-1][0]
        else:
            onion1 = 0
        if len(np.where(obs[...,22:26]>0)[-1]) > 0:
            onion2 = np.where(obs[...,22:26]>0)[-1][0]
        else:
            onion2 = 0
        onions_pot = np.max((onion1, onion2))
        ingredients = []
        for onion in range(onions_pot):
            ingredients.append(ObjectState("onion", pot_loc))

        if np.max(obs[...,29]) == 1:
            serve_loc = (np.nonzero(obs[...,29])[0][0],np.nonzero(obs[...,29])[1][0])
            soup_elapsed = 20
            if serve_loc == p1_loc:
                p1_object = SoupState(serve_loc, ingredients, soup_elapsed, soup_cook_time)
            elif serve_loc == p2_loc:
                p2_object = SoupState(serve_loc, ingredients, soup_elapsed, soup_cook_time)
            else:
                ingredients = []
                for onion in range(onions_pot):
                    ingredients.append(ObjectState("onion", pot_loc))
                if len(ingredients)>0:
                    objects["soup"] = SoupState(serve_loc, ingredients, soup_elapsed, soup_cook_time)
        else:
            soup_loc = pot_loc
            # timer = np.max(obs[...,20])
            if len(np.where(obs[...,28].flatten()>0)[-1]) >0:
                timer = np.where(obs[...,28].flatten()>0)[-1][0]
            else:
                timer = 0
            if timer == 0:
                soup_elapsed = -1
            else:
                soup_elapsed = 20 - timer
            if len(ingredients)>0:
                objects["soup"] = SoupState(soup_loc, ingredients, soup_elapsed, soup_cook_time)
        
        #process other objects
        #onions
        num_onions = np.count_nonzero(obs[...,31])
        for i in range(num_onions):
            onion_loc = (np.nonzero(obs[...,31])[0][i],np.nonzero(obs[...,31])[1][i])
            if onion_loc == p1_loc:
                p1_object = ObjectState("onion", onion_loc)
            elif onion_loc == p2_loc:
                p2_object = ObjectState("onion", onion_loc)
            else:
                objects["onion_{}".format(i)] = ObjectState("onion", onion_loc)
        #dishes
        num_dishes = np.count_nonzero(obs[...,30])
        for i in range(num_dishes):
            dishes_loc = (np.nonzero(obs[...,30])[0][i],np.nonzero(obs[...,30])[1][i])
            if dishes_loc == p1_loc:
                p1_object = ObjectState("dish", dishes_loc)
            elif dishes_loc == p2_loc:
                p2_object = ObjectState("dish", dishes_loc)
            else:
                objects["dish_{}".format(i)] = ObjectState("dish", dishes_loc)
        
        p1 = PlayerState(p1_loc,p1_dir, p1_object)
        p2 = PlayerState(p2_loc,p2_dir, p2_object)

        return [p1,p2], objects

    def render(self, timestep, obs):
        obs = self.overcooked_env.mdp.lossless_state_encoding(obs, debug=False)[0]
        obs = env._preprocess(obs)
        players, objects = self._parse_obs(obs) 
        state = State(players, objects)

        img_directory_path = "temp"
        img_name = "frame_" + str(timestep) + ".jpg"
        img_path = os.path.join(img_directory_path, img_name)
        hud_data = {}
        # hud_data['player1'] = info[1]
        # hud_data['players2'] = info[2]
        # hud_data['rewards'] = info[3]
        # hud_data['shaped_r'] = info[4]
        StateVisualizer().display_rendered_state(state=state, grid=self.overcooked_env.mdp.terrain_mtx, hud_data=hud_data, img_path=img_path, ipython_display=False, window_display=False)

    def process_trajs(self,trajs):
        layout = SCENARIOS[str(self.scenario)]['Layout']
        ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": True}, 
                                            env_params={"horizon": 400})
    
        runs, timesteps = np.array(trajs).shape

        r = []
        t = []
        for run in range(runs):
            t = []
            for step in range(timesteps):
                obs_state = trajs[run][step]['state']
                obs_array = ae.env.mdp.lossless_state_encoding(obs_state, debug=False)[0]
                t.append(obs_array)
            r.append(t)

        out_state = np.array(r)

        r = []
        t = []
        for run in range(runs):
            t = []
            for step in range(timesteps):
                action_pair = np.zeros((2,6))
                actions = trajs[run][step]['action']
                for i in range(2):
                    if actions[i] == "interact":
                        action_pair[i][0] = 1
                    if actions[i] == (0, 0):
                        action_pair[i][1] = 1
                    if actions[i] == (0, -1):
                        action_pair[i][2] = 1
                    if actions[i] == (0, 1):
                        action_pair[i][3] = 1
                    if actions[i] == (1, 0):
                        action_pair[i][4] = 1
                    if actions[i] == (-1, 0):
                        action_pair[i][5] = 1
                t.append(action_pair)
            r.append(t)

        out_action = np.array(r)

        return out_state, out_action
         
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
            # Get the logits, action, and take one step in the environment
            observation = observation.reshape(1, -1)
            logits, action = self.sample_action(observation)
            observation_new, reward, done, info = self.env.step(action[0].numpy())
            reward_sparse, shaped_r = reward
            # rate = np.max((0.0,(1.0-epoch*0.01)))
            rate = 0
            rewards_sum = reward_sparse + shaped_r*rate
            episode_return += rewards_sum 
            episode_length += 1
            replay.append(info)

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

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = self.buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.train_policy_iterations):
            kl = self.train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(self.train_value_iterations):
            self.train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print("Rate:{}".format(rate))
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

        return replay
    
    def train(self, epochs, save):
        reward_hist = []
        reward_hist_s = []        
        p2_prob = []        
        acc_hist = []
        acc_exec_hist = []

        path = save
        if not os.path.exists(path):
            os.makedirs(path)
            
        for i in range(0, epochs, 1):
            print("")
            print("Epoch: {}".format(i))
            replay = self.train_epoch(epoch=i,render=False)
            self.actor.save_weights("{}/actor_{}".format(save, i))

            r_sparse = 0
            r_shape = 0
            acc = 0
            acc_exec = 0
            for t in range(400):
                r_sparse += replay[t][3]
                r_shape += replay[t][4] 
                p2_prob.append(replay[t][5])    
                acc += replay[t][6]       
                acc_exec += replay[t][7]  
            reward_hist.append(r_sparse)
            reward_hist_s.append(r_shape)
            acc_hist.append(acc/400)
            acc_exec_hist.append(acc_exec/400)

            self.actor.save_weights("{}/actor_latest".format(save))    

            np.save("{}/p2_prob".format(save), np.array(p2_prob))
            np.save("{}/reward_hist".format(save), np.array(reward_hist))
            np.save("{}/reward_hist_s".format(save), np.array(reward_hist_s))
            np.save("{}/acc_hist".format(save), np.array(acc_hist))
            np.save("{}/acc_exec_hist".format(save), np.array(acc_exec_hist))

    def eval_ppo(self, suffix):
        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False
        self.actor.load_weights("{}/actor_{}".format(save, suffix))
        agent = env._agent(self.scenario)
        agent_gt = env._agent0(self.scenario)
        sensor = env.Sensor(self.scenario)

        agent_gt.reset()
        agent.reset()
        sensor.reset()

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False

        acc = 0

        r_l = []

        traj = []
        replay = []

        for t in range(4000):
            obs = self.overcooked_env.mdp.lossless_state_encoding(observation, debug=False)[0]
            obs = env._preprocess(obs)
            obs = np.expand_dims(obs,0)

            hidden = sensor.sense(obs,0)

            action1 = self.actor(hidden)
            action1 = self.action_list[np.argmax(tf.keras.layers.Softmax()(action1[0]).numpy())]
            action2 = agent.action(observation)[0]

            action1_gt = np.argmax(agent_gt.action(observation)[1]["action_probs"])
            action1_gt = Action.INDEX_TO_ACTION[action1_gt]

            if action1 == action1_gt:
                acc+= 1
            
            action_pair = [action1,action2]
            observation_new, reward, done, _ = self.overcooked_env.step(action_pair)
            episode_return += reward
            episode_length += 1
            info = {}
            info['state'] = observation
            info['action'] = action_pair
            replay.append(info)
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
                sensor.reset()
                agent_gt.reset()
                agent.reset()

        out_state, out_action = self.process_trajs(traj)
        # Print mean return and length for each epoch
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
        return sum_return/num_episodes, acc/4000, out_state, out_action

    def eval_ppo_traj(self, suffix):
        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False
        self.actor.load_weights("{}/actor_{}".format(save, suffix))
        agent = env._agent(self.scenario)
        sensor = env.Sensor(self.scenario)

        agent.reset()
        sensor.reset()

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False

        r_l = []

        traj = []
        replay = []

        for t in range(400):
            obs = self.overcooked_env.mdp.lossless_state_encoding(observation, debug=False)[0]
            obs = env._preprocess(obs)
            obs = np.expand_dims(obs,0)

            hidden = sensor.sense(obs,0)

            action1 = self.actor(hidden)
            action1 = self.action_list[np.argmax(tf.keras.layers.Softmax()(action1[0]).numpy())]
            action2 = agent.action(observation)[0]

            action_pair = [action1,action2]
            observation_new, reward, done, _ = self.overcooked_env.step(action_pair)
            episode_return += reward
            episode_length += 1
            info = {}
            info['state'] = observation
            info['action'] = action_pair
            replay.append(info)
            observation = observation_new
            self.render(t,observation)
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
                sensor.reset()

        out_state, out_action = self.process_trajs(traj)
        # Print mean return and length for each epoch
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
        return sum_return/num_episodes, out_state, out_action
    
    def generate_trajectories(self, runs):
        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False
        agent = env._agent(self.scenario)
        sensor = env.Sensor(self.scenario)

        agent.reset()
        sensor.reset()

        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done = False

        r_l = []

        traj = []
        replay = []

        for t in range(self.steps_per_epoch*runs):
            obs = self.overcooked_env.mdp.lossless_state_encoding(observation, debug=False)[0]
            obs = env._preprocess(obs)
            obs = np.expand_dims(obs,0)

            hidden = sensor.sense(obs,0)

            action1 = self.actor(hidden)
            action1 = self.action_list[np.argmax(tf.keras.layers.Softmax()(action1[0]).numpy())]
            action2 = agent.action(observation)[0]

            action_pair = [action1,action2]
            observation_new, reward, done, _ = self.overcooked_env.step(action_pair)
            episode_return += reward
            episode_length += 1
            info = {}
            info['state'] = observation
            info['action'] = action_pair
            replay.append(info)
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
                self.critic = self.create_critic()
                self.actor = models._create_behaviour_model()
                observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
                sensor.reset()

        # Print mean return and length for each epoch
        print(
            f"Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

        return traj

def main(epoch, save, scenario, behaviour, temperature):
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    trainer = Trainer(scenario, behaviour, temperature)
    # trainer.train(epoch, save)
    # trainer.eval_ppo_traj(499)

    path = "{}".format(save)
    if not os.path.exists(path):
        os.makedirs(path)

    out_states = None
    out_actions = None

    # with open("{}/res.txt".format(save), "w") as f:
    #     f.close()
    # with open("{}/acc.txt".format(save), "w") as f:
    #     f.close()
    # for suffix in range(0,epoch,1):
    #     if (suffix+1)%10==0 or suffix==0:
    #         res, acc, out_state, out_action = trainer.eval_ppo(suffix)
    #         with open("{}/res.txt".format(save), "a") as f:
    #             f.write(str(res))
    #             f.write('\n')
    #         with open("{}/acc.txt".format(save), "a") as f:
    #             f.write(str(acc))
    #             f.write('\n')
                
    for suffix in range(0,epoch,1):
        if (suffix+1)%5==0 or suffix==0:
            res, out_state, out_action = trainer.eval_ppo_traj(suffix)
            if out_states is None:
                out_states = out_state
                out_actions = out_action
            else:
                out_states = np.concatenate((out_states,out_state))
                out_actions = np.concatenate((out_actions,out_action))

    np.save("data/{}/training_data_f".format(scenario), out_states)
    np.save("data/{}/action_training_data_f".format(scenario), out_actions)

if __name__ == "__main__":  
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    behaviour = "behaviour_best"
    temperature = 0.01
    epoch = 500
    scenario = "CrampedRoom"
    for i in range(1,2,1):
        save = "results/{}/test_seed_{}/{}_{}_{}/".format(scenario,seed,temperature,behaviour,i)
        main(epoch, save, scenario, behaviour, temperature)
    
    # temperature = 1.0
    # epoch = 500
    # scenario = "CrampedRoom"
    # for i in range(3,5,1):
    #     save = "results/{}/test2_rate_0_seed_0/{}_{}_{}/".format(scenario,temperature,behaviour,i)
    #     main(epoch, save, scenario, behaviour, temperature)
    
    # temperature = 0.5
    # epoch = 500
    # scenario = "CrampedRoom"
    # for i in range(1,4,1):
    #     save = "results/{}/test_rate_0/{}_{}_{}/".format(scenario,behaviour,temperature,i)
    #     main(epoch, save, scenario, behaviour, temperature)