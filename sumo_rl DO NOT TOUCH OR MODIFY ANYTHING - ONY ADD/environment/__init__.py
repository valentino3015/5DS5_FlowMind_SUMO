"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register


register(
    id="sumo-rl-v0",
    entry_point="sumo_rl DO NOT TOUCH OR MODIFY ANYTHING - ONY ADD.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
