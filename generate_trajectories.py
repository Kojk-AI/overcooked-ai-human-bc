
import numpy as np
import os

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

from human_aware_rl.rllib.rllib import load_agent_pair, AgentPair, load_agent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import train

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
    "ForcedCoordiantion":{
        "Agent": "RllibForcedCoordinationSP",
        "Layout": "forced_coordination"
    },
}

def gen_trajs_ppo(runs, sp, ae):
    trajs = ae.evaluate_agent_pair(sp,runs,400)

    runs, timesteps = trajs['ep_states'].shape

    #parse state observations from trajectories
    r = []
    t = []
    for run in range(runs):
        t = []
        for step in range(timesteps):
            obs_state = trajs['ep_states'][run][step]
            obs_array = ae.env.mdp.lossless_state_encoding(obs_state, debug=False)[0]
            t.append(obs_array)
        r.append(t)
    out_state = np.array(r)

    #parse agents actions from trajectories
    r = []
    t = []
    for run in range(runs):
        t = []
        for step in range(timesteps):
            action_pair = np.zeros((2,6))
            actions = trajs['ep_actions'][run][step]
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

def gen_trajs_random(trajs, ae):
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

def generate_trajectories(scenario, runs):
    agent_path = "src/overcooked_demo/server/static/assets/agents/{}/agent".format(SCENARIOS[str(scenario)]['Agent'])
    sp = load_agent_pair(agent_path,"ppo","ppo")

    layout = SCENARIOS[str(scenario)]['Layout']
    ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": True}, 
                                        env_params={"horizon": 400})

    trainer = train.Trainer(scenario)
    trajs_random = trainer.generate_trajectories(runs)

    out_state_r, out_action_r = gen_trajs_random(trajs_random, ae)
    out_state, out_action = gen_trajs_ppo(runs, sp, ae)

    return out_state, out_action, out_state_r, out_action_r


def main(scenario):
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    out_state, out_action, out_state_r, out_action_r = generate_trajectories(scenario, 100)

    path = "data/{}".format(scenario)
    if not os.path.exists(path):
        os.makedirs(path)

    np.save("data/{}/training_data".format(scenario), out_state)
    np.save("data/{}/action_training_data".format(scenario), out_action)

    np.save("data/{}/training_data_r".format(scenario), out_state_r)
    np.save("data/{}/action_training_data_r".format(scenario), out_action_r)

if __name__ == "__main__":
    scenario = "CoordinationRing"
    main(scenario)