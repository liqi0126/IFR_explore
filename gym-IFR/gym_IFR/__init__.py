import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='IFR-v0',
    entry_point='gym_IFR.envs:IFR',
    max_episode_steps=500,
    reward_threshold=1000,
    nondeterministic=True,
)

register(
    id='IFR_eval-v0',
    entry_point='gym_IFR.envs:IFREval',
    max_episode_steps=500,
    reward_threshold=1000,
    nondeterministic=True,
)
