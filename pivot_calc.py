import urllib.request
import csv
from datetime import datetime

URL = 'https://stooq.com/q/d/l/?s=goog.us&i=d'


def fetch_data(url=URL):
    with urllib.request.urlopen(url) as resp:
        lines = resp.read().decode('utf-8').splitlines()
    reader = csv.DictReader(lines)
    data = [row for row in reader]
    return data

def compute_pivot(data, start_date, end_date, pip=0.01):
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    highs, lows, closes = [], [], []
    for row in data:
        d = datetime.fromisoformat(row['Date'])
        if start <= d <= end:
            highs.append(float(row['High']))
            lows.append(float(row['Low']))
            closes.append(float(row['Close']))
    if not highs:
        raise ValueError('No data in range')
    H = max(highs)
    L = min(lows)
    C = closes[-1]
    pp = (H + L + C) / 3
    r1 = 2 * pp - L
    s1 = 2 * pp - H
    return {
        'pivot': round(pp, 4),
        'support': round(s1 - 50 * pip, 4),
        'resistance': round(r1 + 50 * pip, 4)
    }

if __name__ == '__main__':
    data = fetch_data()
    levels = compute_pivot(data, '2024-03-26', '2024-04-10')
    print(levels)
