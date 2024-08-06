import tensorflow as tf
import numpy as np
import os

import models
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, ObjectState, SoupState
from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import overcooked
from overcooked_ai_py.mdp.actions import Action
from human_aware_rl.rllib.rllib import load_agent
from human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model
from human_aware_rl.imitation.behavior_cloning_tf2 import _get_base_ae, BehaviorCloningPolicy
from human_aware_rl.rllib.rllib import RlLibAgent

SCENARIOS = {
    "CrampedRoom":{
        "Agent": "RllibCrampedRoomSP",
        "Layout": "cramped_room",
        "Layout_size": "54",
    },
    "AsymmetricAdvantages":{
        "Agent": "RllibAsymmetricAdvantagesSP",
        "Layout": "asymmetric_advantages",
        "Layout_size": "95",
    },
    "CoordinationRing":{
        "Agent": "RllibCoordinationRingSP",
        "Layout": "coordination_ring",
        "Layout_size": "55",
    },
    "ForcedCoordination":{
        "Agent": "RllibForcedCoordinationSP",
        "Layout": "forced_coordination",
        "Layout_size": "55",
    },
    "CounterCircuit":{
        "Agent": "RllibCounterCircuit1OrderSP",
        "Layout": "counter_circuit_o_1order",
        "Layout_size": "85",
    },
}

NUM_HIDDEN_LAYERS = 3
SIZE_HIDDEN_LAYERS = 64
NUM_FILTERS = 25
NUM_CONV_LAYERS = 3

CELL_SIZE = 256

# GRID = [['X', 'X', 'P', 'X', 'X'], ['O', ' ', ' ', ' ', 'O'], ['X', ' ', ' ', ' ', 'X'], ['X', 'D', 'X', 'S', 'X']]

def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T

def _agent(scenario):
    agent_path = "src/overcooked_demo/server/static/assets/agents/{}/agent".format(SCENARIOS[str(scenario)]['Agent'])
    ppo_agent = load_agent(agent_path,"ppo",1)
    return ppo_agent

def _agent0(scenario):
    agent_path = "src/overcooked_demo/server/static/assets/agents/{}/agent".format(SCENARIOS[str(scenario)]['Agent'])
    ppo_agent = load_agent(agent_path,"ppo",0)
    return ppo_agent

# def _bc_agent():
#     # bc_model_path = "bc_dir_cramped_room"
#     bc_model_path = "bc_runs/train/cramped_room"
#     bc_model, bc_params = load_bc_model(bc_model_path)

#     bc_policy = BehaviorCloningPolicy.from_model(
#             bc_model, bc_params, stochastic=True
#         )
#     # We need the featurization function that is specifically defined for BC agent
#     # The easiest way to do it is to create a base environment from the configuration and extract the featurization function
#     # The environment is also needed to do evaluation

#     base_ae = _get_base_ae(bc_params)
#     base_env = base_ae.env

#     bc_agent = RlLibAgent(bc_policy,1,base_env.featurize_state_mdp)
#     return bc_agent

def _preprocess(obs):        
    ph = np.zeros((obs.shape[0],obs.shape[1],obs.shape[2]+8))
    for layer in range(16):
        ph[...,layer] = obs[...,layer]
    for layer in range(16,19,1):
        if layer == 16:
            ph[...,layer+int(np.max(obs[...,layer]))] = np.where(obs[...,layer]>0,1,0)
        if layer == 17:
            ph[...,layer+4] = obs[...,layer]
        if layer == 18:
            ph[...,layer+int(np.max(obs[...,layer]))+4] = np.where(obs[...,layer]>0,1,0)
    for layer in range(19,26,1):
        if layer==20:
            temp = np.zeros(obs.shape[0]*obs.shape[1])
            timer = np.max(obs[...,layer])
            temp[timer] = 1
            temp = np.reshape(temp, (obs.shape[0],obs.shape[1]))
            ph[...,layer+8] = temp
        else:
            ph[...,layer+8] = obs[...,layer]         

    return ph

def _parse_obs(obs):
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

class State:
    def __init__(self, players, objects=None):
        self.players = players
        self.objects = objects

class Sensor:
    def __init__(self, scenario, behaviour="behaviour_best", temperature=1.0):
        self.lstm = models._create_lstm()
        self.lstm.load_weights("data/{}/pretrain/lstm".format(scenario))
        self.lstm2 = models._create_lstm()
        self.lstm.load_weights("data/{}/pretrain/lstm".format(scenario))
        layout_size = (int(tuple(SCENARIOS[scenario]['Layout_size'])[0]),int(tuple(SCENARIOS[scenario]['Layout_size'])[1]))
        self.enc = models._create_enc(layout_size)
        self.enc.load_weights("data/{}/pretrain/encoder".format(scenario))
        self.behaviour = models._create_behaviour_model()
        self.behaviour.load_weights("data/{}/behaviour/{}".format(scenario, behaviour))

    def reset(self):
        self.h_in = np.zeros((1,CELL_SIZE))
        self.c_in = np.zeros((1,CELL_SIZE))
        self.h_in2 = np.zeros((1,CELL_SIZE))
        self.c_in2 = np.zeros((1,CELL_SIZE))

    def sense(self,obs,id):
        obs[...,33] = 0
        out = self.enc(obs)
        out = np.expand_dims(out,0)
        if id == 0:
            out, self.h_in, self.c_in = self.lstm([out, self.h_in, self.c_in])
        if id == 1:
            out, self.h_in2, self.c_in2 = self.lstm([out, self.h_in2, self.c_in2])
        return out
    
    def action(self,lstm_state):
        p2_action = self.behaviour(lstm_state)
        action2 = int(tf.random.categorical(p2_action, 1).numpy())
        # action2 = np.argmax(p2_action)
        # p2_action = np.zeros((6))
        # p2_action[action2] = 1
        return action2

class Env:
    def __init__(self, scenario, behaviour, temperature):
        self.lstm = models._create_lstm()
        self.lstm.load_weights("data/{}/pretrain/lstm".format(scenario))
        self.lstm2 = models._create_lstm()
        self.lstm2.load_weights("data/{}/pretrain/lstm".format(scenario))
        layout_size = (int(tuple(SCENARIOS[scenario]['Layout_size'])[0]),int(tuple(SCENARIOS[scenario]['Layout_size'])[1]))
        self.enc = models._create_enc(layout_size)
        self.enc.load_weights("data/{}/pretrain/encoder".format(scenario))
        self.behaviour = models._create_behaviour_model(temperature=temperature)
        self.behaviour.load_weights("data/{}/behaviour/{}".format(scenario, behaviour))

        self.action_list = [Action.INTERACT,Action.STAY,Action.INDEX_TO_ACTION[0],Action.INDEX_TO_ACTION[1],Action.INDEX_TO_ACTION[2],Action.INDEX_TO_ACTION[3]]
        self.overcooked_env = overcooked.Overcooked(scenario)

        self.agent = _agent(scenario)

    def reset(self):
        self.h_in = np.zeros((1,CELL_SIZE))
        self.c_in = np.zeros((1,CELL_SIZE))
        self.h_in2 = np.zeros((1,CELL_SIZE))
        self.c_in2 = np.zeros((1,CELL_SIZE))
        # self.state = INITIAL_STATE
        self.agent.reset()
        # self.state = _preprocess(self.state)

        self.counter = 0

        self.observation, episode_return, episode_length = self.overcooked_env.reset(), 0, 0
        self.state = _preprocess(self.overcooked_env.mdp.lossless_state_encoding(self.observation, debug=False)[0])
        state = self.enc(np.expand_dims(self.state,0))
        state = np.expand_dims(state,0)
        lstm_state, h, c, = self.lstm([state,self.h_in,self.c_in])

        return lstm_state

    def step(self,action):
        current_state = self.state

        state_2 = current_state.copy()
        state_2[...,0] = current_state[...,1]
        state_2[...,1] = current_state[...,0]
        state_2[...,2] = current_state[...,6]
        state_2[...,3] = current_state[...,7]
        state_2[...,4] = current_state[...,8]
        state_2[...,5] = current_state[...,9]
        state_2[...,6] = current_state[...,2]
        state_2[...,7] = current_state[...,3]
        state_2[...,8] = current_state[...,4]
        state_2[...,9] = current_state[...,5]
        state_2[...,33] = 0

        # self.state = self.enc(np.expand_dims(self.state,0))
        state_2 = self.enc(np.expand_dims(state_2,0))
        # self.state = np.expand_dims(self.state,0)
        state_2 = np.expand_dims(state_2,0)

        lstm_state2, h2, c2, = self.lstm2([state_2,self.h_in2,self.c_in2])

        p1_action = np.zeros((1,6))
        p1_action[0,action] = 1
        p1 = self.action_list[action]

        bc_gt = -1
        bc_gt_exec = -1

        use_agent = True

        if use_agent:
            p2_action = self.agent.action(self.observation)[0]
            p2 = p2_action
            p2_action_probs = self.agent.action(self.observation)[1]["action_probs"]

            action_gt = np.argmax(p2_action_probs)
            action_gt = Action.INDEX_TO_ACTION[action_gt]

            if p2 == action_gt:
                bc_gt_exec = 1
            else:
                bc_gt_exec = 0

        else:
            p2_action_probs = self.behaviour(lstm_state2)
            # action2_prob = np.max(p2_action_probs)
            action2_prob = tf.math.log(0.8)
            # action2_prob = tf.math.log(0.8)
            action2_prob_complement = tf.math.log((1.0-0.8)/5)
            p2_action_random = np.zeros((1,6))
            for i in range(6):
                if i == np.argmax(p2_action_probs):
                    p2_action_random[0,i] = action2_prob
                else:
                    p2_action_random[0,i] = action2_prob_complement

            action2 = int(tf.random.categorical(p2_action_random, 1).numpy())
            p2_action_probs = softmax(p2_action_probs)

            # action2 = np.argmax(p2_action_probs)
            p2_action = np.zeros((1,6))
            p2_action[0,action2] = 1
            p2 = self.action_list[action2]

            action_bc = np.argmax(p2_action_probs)
            action_bc = self.action_list[action_bc]
            action_gt = np.argmax(self.agent.action(self.observation)[1]["action_probs"])
            action_gt = Action.INDEX_TO_ACTION[action_gt]
            # p2_gt = self.agent.action(self.observation)[0]

            if action_bc == action_gt:
                bc_gt = 1
            else:
                bc_gt = 0

            if p2 == action_gt:
                bc_gt_exec = 1
            else:
                bc_gt_exec = 0

        action_pair = [p1,p2]
        observation_new, reward, done, env_info = self.overcooked_env.step(action_pair)
        shaped_r = sum(env_info['shaped_r_by_agent'])
        next_state = _preprocess(self.overcooked_env.mdp.lossless_state_encoding(observation_new, debug=False)[0])
        rewards_sum = reward+shaped_r

        self.state = next_state
        self.counter += 1

        if self.counter >= 400:
            done = True
        else:
            done = False

        lstm_state = self.enc(np.expand_dims(self.state,0))
        lstm_state = np.expand_dims(lstm_state,0)
        lstm_state, h, c, = self.lstm([lstm_state,self.h_in,self.c_in])

        self.h_in = h
        self.c_in = c
        self.h_in2 = h2
        self.c_in2 = c2

        self.observation = observation_new
        return lstm_state, (reward,shaped_r), done, [current_state,p1_action,p2_action,reward,shaped_r,p2_action_probs,bc_gt,bc_gt_exec]

    # def render(self, timestep, obs):
    #     players, objects = _parse_obs(obs) 
    #     state = State(players, objects)

    #     img_directory_path = "temp"
    #     img_name = "frame_" + str(timestep) + ".jpg"
    #     img_path = os.path.join(img_directory_path, img_name)
    #     hud_data = {}
    #     # hud_data['player1'] = info[1]
    #     # hud_data['players2'] = info[2]
    #     # hud_data['rewards'] = info[3]
    #     # hud_data['shaped_r'] = info[4]
    #     StateVisualizer().display_rendered_state(state=state, grid=GRID, hud_data=hud_data, img_path=img_path, ipython_display=False, window_display=False)