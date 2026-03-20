from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv
from toy_mp.envs.conveyor_portal.map_spec import MapSpec
from toy_mp.rllib.multiagent_wrapper import SequentialPhaseMAEnv


def test_wrapper_turn_based_step():
    spec = MapSpec.model_validate(
        {
            "width": 6,
            "height": 6,
            "start": [1, 1],
            "goal": [5, 5],
            "belts": [{"id": "b1", "path": [[2, 1], [3, 1], [4, 1]]}],
            "portals": [{"src": [3, 0], "dst": [4, 4]}],
            "wait_n_max": 3,
            "max_steps": 30,
        }
    )
    ma = SequentialPhaseMAEnv(ConveyorPortalEnv(spec))
    obs, infos = ma.reset(seed=0)
    assert list(obs.keys()) == ["phase1"]

    obs, rew, term, trunc, infos = ma.step({"phase1": 1})
    assert "__all__" in term and "__all__" in trunc
    assert not term["__all__"]
    assert "phase2" in obs
