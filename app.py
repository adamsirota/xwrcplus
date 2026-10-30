"""Flask entry point: python -m flask --app app run."""
from pathlib import Path
from mlb_xwrc.web import create_app

app = create_app(Path(__file__).with_name('config.json'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
