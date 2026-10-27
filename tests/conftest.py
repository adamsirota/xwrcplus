import numpy as np
import pandas as pd
import pytest
from mlb_xwrc.data import FEATURES


@pytest.fixture
def panel():
    rng = np.random.default_rng(42)
    rows = []
    for player in range(1,41):
        talent = rng.normal(0, .025)
        for year in range(2016,2027):
            bb = np.clip(.09+talent+rng.normal(0,.01),.02,.2)
            k = np.clip(.22-talent+rng.normal(0,.02),.05,.4)
            ev = 88 + talent*80+rng.normal(0,1)
            age = 20 + player % 12 + year-2016
            rows.append(dict(player_id=player, season=year, name=f'TEST PLAYER {player}',
                age=age, pa=500, bb_rate=bb, k_rate=k, o_swing_rate=.3,
                contact_rate=.8, swstr_rate=.1, exit_velocity=ev, launch_angle=12,
                barrel_rate=.08, hard_hit_rate=.4,
                wrc_plus=100+600*(bb-.09)-180*(k-.22)+2*(ev-88)-max(age-30,0)+rng.normal(0,10)))
    return pd.DataFrame(rows)
