from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv
from toy_mp.envs.conveyor_portal.map_spec import MapSpec


def test_smoke_imports_and_reset():
    spec = MapSpec.model_validate(
        {
            "width": 5,
            "height": 5,
            "start": [1, 1],
            "goal": [3, 3],
            "belts": [],
            "portals": [],
        }
    )
    env = ConveyorPortalEnv(spec)
    obs, info = env.reset(seed=123)
    assert obs["phase"] == 1
    assert info["phase"] == 1
