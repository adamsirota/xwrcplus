"""Run with python -m mlb_xwrc {download,run} --config config.json."""
import argparse
import json
import sys
from pathlib import Path
from .data import download


def main():
    parser = argparse.ArgumentParser(description='Expected MLB hitting and next-season aging projections')
    parser.add_argument('command',choices=['download','run'])
    parser.add_argument('--config',default='config.json')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    try:
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        if config['application_season']-config['start_season'] < 8:
            raise ValueError('At least eight historical seasons are required for chronological forecast evaluation.')
        if config['min_train_pa'] <= 0 or config['min_rank_pa'] < config['min_train_pa']:
            raise ValueError('Require 0 < min_train_pa <= min_rank_pa.')
        if args.command=='download':
            errors = download(config,config_path.parent)
            return 1 if errors else 0
        from .pipeline import run_analysis
        run_analysis(config,config_path.parent)
    except (ValueError,KeyError,OSError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
