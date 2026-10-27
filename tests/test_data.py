import pandas as pd
import pytest
from mlb_xwrc.data import normalize_batting, join_contact, validate_panel


def test_duplicate_team_stints_require_season_total():
    raw = pd.DataFrame({'IDfg':[1,1], 'Season':[2025,2025], 'Name':['A','A'],
                        'Team':['NYY','BOS'], 'Age':[27,27], 'PA':[200,300], 'wRC+':[100,120]})
    with pytest.raises(ValueError, match='total'):
        normalize_batting(raw)
    total = raw.iloc[[0]].assign(Team='2 Tms', PA=500, **{'wRC+':112})
    got = normalize_batting(pd.concat([raw,total]))
    assert len(got) == 1
    assert got.iloc[0].wrc_plus == 112


def test_id_join_does_not_match_names():
    fg = pd.DataFrame({'player_id':[1,2], 'season':[2025,2025], 'name':['Same','Same']})
    mapping = pd.DataFrame({'key_fangraphs':[1,2], 'key_mlbam':[10,20]})
    contact = pd.DataFrame({'player_id':[20], 'season':[2025], 'avg_hit_speed':[95]})
    got = join_contact(fg, contact, mapping)
    assert pd.isna(got.iloc[0].exit_velocity)
    assert got.iloc[1].exit_velocity == 95


def test_percent_strings_and_invalid_units():
    raw = pd.DataFrame({'IDfg':[1], 'Season':[2025], 'Name':['A'], 'Age':[27],
                        'PA':[500], 'wRC+':[120], 'BB%':['10.0%'], 'K%':[0.2]})
    got = normalize_batting(raw)
    assert got.iloc[0].bb_rate == .1
    with pytest.raises(ValueError, match='bb_rate'):
        validate_panel(got.assign(bb_rate=10))

def test_modern_api_preserves_ids_names_and_totals(monkeypatch, tmp_path):
    from mlb_xwrc.data import fetch_fangraphs_current
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'totalCount':1, 'data':[{'playerid':12,'PlayerName':'A & B','Name':'<a>A &amp; B</a>',
                'Team':'<a>2 Tms</a>','Season':2025,'Age':29,'PA':500,'wRC+':110,'BB%':.1,'K%':.2}]}
    monkeypatch.setattr('requests.get',lambda *args,**kwargs:Response())
    got = fetch_fangraphs_current(2025,2025,tmp_path/'response.json')
    assert got.iloc[0].IDfg == 12
    assert got.iloc[0].Name == 'A & B'
    assert got.iloc[0].Team == '2 Tms'
    assert (tmp_path/'response.page1.json').exists()

def test_multiple_unmapped_ids_are_retained_without_false_matches():
    fg = pd.DataFrame({'player_id':[1,2,3], 'season':[2025]*3})
    mapping = pd.DataFrame({'key_fangraphs':[1], 'key_mlbam':[10]})
    contact = pd.DataFrame({'player_id':[10], 'season':[2025], 'avg_hit_speed':[90]})
    got = join_contact(fg,contact,mapping)
    assert len(got)==3
    assert got.exit_velocity.notna().sum()==1

def test_low_sample_contact_velocity_can_be_below_40_mph():
    frame = pd.DataFrame({'player_id':[1], 'season':[2025], 'name':['A'], 'age':[25],
                          'pa':[1], 'wrc_plus':[0], 'exit_velocity':[24.5]})
    validate_panel(frame)
