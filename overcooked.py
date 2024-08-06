from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

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

def load_mdp(scenario):
    mdp = OvercookedGridworld.from_layout_name(SCENARIOS[scenario]['Layout'])
    print(mdp)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=400)
    return mdp, base_env

class Overcooked():
    def __init__(self, scenario):
        self.mdp, self.base = load_mdp(scenario)

    def reset(self):
        self.base.reset()

        return self.base.state

    def step(self,action_pair):
        action_1 = action_pair[0]
        action_2 = action_pair[1]
        next_state, timestep_sparse_reward, done, env_info = self.base.step([action_1,action_2])

        return self.base.state, timestep_sparse_reward, done, env_info

