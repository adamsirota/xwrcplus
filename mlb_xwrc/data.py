"""Source normalization, audited identifier joins, and resumable pybaseball downloads."""
import json
import html
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ['bb_rate', 'k_rate', 'o_swing_rate', 'contact_rate', 'swstr_rate']
CONTACT = ['exit_velocity', 'launch_angle', 'barrel_rate', 'hard_hit_rate']
FEATURES = CORE + CONTACT
ALIASES = {'IDfg':'player_id', 'Season':'season', 'Name':'name', 'Age':'age',
           'PA':'pa', 'wRC+':'wrc_plus', 'BB%':'bb_rate', 'K%':'k_rate',
           'O-Swing%':'o_swing_rate', 'Contact%':'contact_rate', 'SwStr%':'swstr_rate'}
CONTACT_ALIASES = {'avg_hit_speed':'exit_velocity', 'avg_hit_angle':'launch_angle',
                   'brl_percent':'barrel_rate', 'ev95percent':'hard_hit_rate'}


def numeric(series):
    text = series.astype(str).str.strip()
    values = pd.to_numeric(text.str.rstrip('%'), errors='coerce')
    return values.where(~text.str.endswith('%'), values / 100)


def normalize_batting(raw):
    frame = raw.rename(columns=ALIASES).copy()
    required = ['player_id', 'season', 'name', 'age', 'pa', 'wrc_plus']
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Batting CSV missing columns: {sorted(missing)}')
    keys = ['player_id', 'season']
    if frame.duplicated(keys).any():
        keep = []
        for _, group in frame.groupby(keys, dropna=False):
            if len(group) > 1:
                totals = group[group.get('Team', pd.Series('', index=group.index))
                               .astype(str).str.contains(r'Tms|TOT|---', regex=True)]
                if len(totals) != 1:
                    raise ValueError('Duplicate player-year requires one source season total; do not average wRC+.')
                group = totals
            keep.append(group)
        frame = pd.concat(keep, ignore_index=True)
    for col in required[:2] + ['age', 'pa', 'wrc_plus'] + FEATURES:
        if col in frame:
            frame[col] = numeric(frame[col])
    frame = frame[required + [col for col in FEATURES if col in frame]].copy()
    validate_panel(frame)
    return frame.sort_values(keys).reset_index(drop=True)


def validate_panel(frame):
    required = ['player_id', 'season', 'age', 'pa', 'wrc_plus']
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Empty panel or missing player IDs, seasons, age, PA, or wRC+.')
    if frame.duplicated(['player_id','season']).any():
        raise ValueError('Duplicate player-year.')
    if (frame.player_id <= 0).any() or (frame.pa <= 0).any():
        raise ValueError('Player IDs and PA must be positive.')
    if not frame.age.between(16, 55).all():
        raise ValueError('Age outside supported 16–55 range.')
    for col in ['player_id','season']:
        if not (frame[col] % 1 == 0).all():
            raise ValueError(f'{col} must be integer-valued.')
    for col in CORE + ['barrel_rate','hard_hit_rate']:
        if col in frame and not frame[col].dropna().between(0, 1).all():
            raise ValueError(f'{col} must be a fraction between 0 and 1.')
    for col, low, high in [('exit_velocity',0,125), ('launch_angle',-90,90)]:
        if col in frame and not frame[col].dropna().between(low,high).all():
            raise ValueError(f'{col} outside expected units/range.')
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite required values.')


def join_contact(batting, contact, mapping):
    mapping = mapping[['key_fangraphs','key_mlbam']].apply(pd.to_numeric, errors='coerce')
    mapping = mapping.dropna().query('key_fangraphs > 0 and key_mlbam > 0').drop_duplicates()
    if mapping.key_fangraphs.duplicated().any() or mapping.key_mlbam.duplicated().any():
        raise ValueError('Ambiguous player-ID mapping.')
    source = contact.rename(columns={'player_id':'mlbam_id', **CONTACT_ALIASES}).copy()
    for col in ['mlbam_id','season'] + CONTACT:
        if col in source:
            source[col] = pd.to_numeric(source[col], errors='coerce')
    for col in ['barrel_rate','hard_hit_rate']:
        if col in source:
            source[col] = source[col] / 100  # Savant leaderboard rates are percentage points.
    if source[['mlbam_id','season']].isna().any().any() or (source.mlbam_id <= 0).any():
        raise ValueError('Invalid Statcast player IDs or seasons.')
    columns = ['mlbam_id','season'] + [c for c in CONTACT if c in source]
    if source.duplicated(['mlbam_id','season']).any():
        raise ValueError('Duplicate Statcast player-year.')
    result = batting.merge(mapping.rename(columns={'key_fangraphs':'player_id',
                           'key_mlbam':'mlbam_id'}), on='player_id', how='left', validate='many_to_one')
    result = result.merge(source[columns], on=['mlbam_id','season'], how='left', validate='many_to_one')
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def fetch_fangraphs_current(start, end, response_path, players=None):
    """Read the current public leaderboard API when the legacy pybaseball route fails."""
    import requests
    rows, page = [], 1
    while True:
        params = dict(age='', pos='all', stats='bat', lg='all', qual=1,
                      season=end, season1=start, ind=1, type=8, pageitems=2000, pagenum=page)
        if players:
            params['players'] = ','.join(str(int(p)) for p in players)
        response = requests.get('https://www.fangraphs.com/api/leaders/major-league/data',
                                params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        write_json(Path(response_path).with_suffix(f'.page{page}.json'), payload)
        batch = payload.get('data', [])
        if not batch:
            raise ValueError('Current FanGraphs API returned an empty or incomplete page.')
        rows.extend(batch)
        if len(rows) >= int(payload['totalCount']):
            break
        page += 1
        if page > 100:
            raise ValueError('Unexpected FanGraphs pagination; refusing an unbounded download.')
    frame = pd.DataFrame(rows).rename(columns={'playerid':'IDfg'})
    strip_html = lambda value: html.unescape(re.sub(r'<[^>]+>', '', str(value)))
    for col in ['Name', 'Team']:
        frame[col] = frame[col].map(strip_html)
    if 'PlayerName' in frame:
        frame['Name'] = frame.PlayerName.fillna(frame.Name)
    if not frame.Season.between(start,end).all():
        raise ValueError('FanGraphs returned seasons outside the requested range.')
    if players and not set(frame.IDfg).issubset(set(players)):
        raise ValueError('FanGraphs did not honor the player filter.')
    return frame


def download(config, root):
    """Cache provider tables with metadata; interrupted requests time out after 60 seconds."""
    import requests
    import pybaseball as pb
    from unittest.mock import patch

    raw = Path(root) / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    # pybaseball 2.2.7 has requests without a timeout. Bound those calls locally.
    original = requests.sessions.Session.request
    def bounded(session, method, url, **kwargs):
        kwargs['timeout'] = kwargs.get('timeout') or 60
        return original(session, method, url, **kwargs)

    errors = []
    def cached(filename, source, fetch):
        path = raw / filename
        if path.exists():
            return pd.read_csv(path, low_memory=False)
        print(f'Downloading {filename}', flush=True)
        try:
            fallback_reason = None
            try:
                frame = fetch()
            except Exception as exc:
                if not (filename.startswith('batting_') or filename == 'career_history.csv'):
                    raise
                fallback_reason = str(exc)
                if filename == 'career_history.csv':
                    start, end = config['history_start'], config['start_season']-1
                    latest = pd.read_csv(raw/f'batting_{config["application_season"]}.csv')
                    players = latest.loc[latest.PA >= config['min_train_pa'], 'IDfg'].astype(int).tolist()
                else:
                    start = end = int(filename.removeprefix('batting_').removesuffix('.csv'))
                    players = None
                frame = fetch_fangraphs_current(start, end, path.with_suffix('.json'), players)
                source = 'FanGraphs current public leaderboard API (legacy pybaseball fallback)'
            if frame.empty:
                raise ValueError('Provider returned an empty table.')
            frame.to_csv(path, index=False)
            write_json(path.with_suffix('.meta.json'), {'source':source,
                       'retrieved_utc':datetime.now(timezone.utc).isoformat(), 'rows':len(frame),
                       'legacy_fallback_reason':fallback_reason})
            return frame
        except Exception as exc:
            errors.append({'file':filename, 'error':str(exc)})
            return None

    with patch.object(requests.sessions.Session, 'request', bounded):
        for year in range(config['start_season'], config['application_season'] + 1):
            cached(f'batting_{year}.csv', 'pybaseball.batting_stats (FanGraphs, regular season)',
                   lambda y=year: pb.batting_stats(y, y, qual=1, ind=1))
            cached(f'contact_{year}.csv', 'pybaseball.statcast_batter_exitvelo_barrels (Savant)',
                   lambda y=year: pb.statcast_batter_exitvelo_barrels(y, minBBE=1))
        cached('player_ids.csv', 'pybaseball.chadwick_register', lambda:pb.chadwick_register())
        cached('career_history.csv', 'pybaseball.batting_stats, historical regular seasons',
               lambda:pb.batting_stats(config['history_start'], config['start_season']-1, qual=1, ind=1))
    write_json(raw / 'download_errors.json', errors)
    if errors:
        print(f'{len(errors)} download(s) failed; see {raw / "download_errors.json"}', flush=True)
    return errors


def load_panel(config, root):
    root = Path(root)
    if config.get('input_csv'):
        panel = normalize_batting(pd.read_csv(root / config['input_csv']))
        for col in FEATURES:
            if col not in panel:
                panel[col] = np.nan
        validate_panel(panel)
        return panel, {'source':'local CSV (user-supplied provenance)', 'contact_join':'provided in CSV'}
    raw = root / 'data/raw'
    expected = list(range(config['start_season'], config['application_season']+1))
    missing = [y for y in expected if not (raw / f'batting_{y}.csv').exists()]
    if missing:
        raise ValueError(f'Missing batting seasons {missing}. Run download or configure input_csv. See data/raw/download_errors.json.')
    batting = pd.concat([normalize_batting(pd.read_csv(raw / f'batting_{y}.csv')) for y in expected], ignore_index=True)
    contact_parts = []
    for year in expected:
        path = raw / f'contact_{year}.csv'
        if path.exists():
            contact_parts.append(pd.read_csv(path).assign(season=year))
    note = {'source':'FanGraphs (pybaseball or documented current-API fallback) + pybaseball/Baseball Savant', 'contact_join':'unavailable; discipline-only fallback'}
    if contact_parts and (raw / 'player_ids.csv').exists():
        batting = join_contact(batting, pd.concat(contact_parts), pd.read_csv(raw/'player_ids.csv', low_memory=False))
        note['contact_join'] = 'stable FanGraphs → MLBAM identifiers; unmatched rows retain missing values'
    for col in FEATURES:
        if col not in batting:
            batting[col] = np.nan
    validate_panel(batting)
    return batting, note


def load_history(panel, root):
    path = Path(root) / 'data/raw/career_history.csv'
    if path.exists():
        earlier = normalize_batting(pd.read_csv(path, low_memory=False))
        earlier = earlier[earlier.season < panel.season.min()]
        return pd.concat([earlier, panel], ignore_index=True).sort_values(['player_id','season'])
    return panel.copy()
