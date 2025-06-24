# Create a reference resampling script so you can inspect / modify it
import textwrap, pathlib, json, datetime, pytz, os, pandas as pd

code = textwrap.dedent("""
    \"\"\"zanzibar_resample.py
    Lightweight resampler that converts freshly appended M1 CSV files into
    M5, M15, M30, H1, H4, H12, Daily, and Weekly OHLCV datasets.
    The automation task (every 15 minutes) just calls `resample_latest()`.
    \"\"\"

    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    DATA_ROOT = Path('/mnt/data')
    M1_DIR    = DATA_ROOT / 'm1_data'      # drop new M1 CSVs here
    OUT_DIR   = DATA_ROOT / 'resampled'    # resampled outputs

    TF_RULES = {
        'M5':  '5T',
        'M15': '15T',
        'M30': '30T',
        'H1':  '1H',
        'H4':  '4H',
        'H12': '12H',
        'D1':  '1D',
        'W1':  '1W'
    }

    def _clean_df(df):
        df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
        df = df.set_index('datetime')[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']]
        df.columns = ['open','high','low','close','tickvol']
        return df

    def _resample(df, rule):
        return df.resample(rule).agg({
            'open'    : 'first',
            'high'    : 'max',
            'low'     : 'min',
            'close'   : 'last',
            'tickvol' : 'sum'
        }).dropna()

    def resample_file(path: Path):
        df = _clean_df(pd.read_csv(path, sep='\\t'))
        symbol = path.stem.split('_')[0]   # 'EURUSD_M1_...csv' -> 'EURUSD'
        for label, rule in TF_RULES.items():
            out_folder = OUT_DIR / symbol
            out_folder.mkdir(parents=True, exist_ok=True)
            _resample(df, rule).to_parquet(out_folder / f\"{symbol}_{label}.parquet\" )

    def resample_latest():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        latest_files = sorted(M1_DIR.glob('*_M1_*.csv'))[-50:]   # scan last 50 for speed
        for file in latest_files:
            resample_file(file)

    if __name__ == '__main__':
        resample_latest()
    """)

file_path = pathlib.Path('/mnt/data/zanzibar_resample.py')
with open(file_path, 'w') as f:
    f.write(code)

file_path
