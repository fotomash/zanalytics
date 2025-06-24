
# entry_logic.py

def validate_structure_shift(data, swing_index, direction='bullish'):
    if direction == 'bullish':
        return data[swing_index]['close'] > data[swing_index - 1]['high']
    else:
        return data[swing_index]['close'] < data[swing_index - 1]['low']

def find_fvg(data):
    fvg_list = []
    for i in range(2, len(data)):
        high0 = data[i - 2]['high']
        low2 = data[i]['low']
        if high0 < low2:
            fvg_list.append({'start': high0, 'end': low2, 'type': 'bullish'})
        low0 = data[i - 2]['low']
        high2 = data[i]['high']
        if low0 > high2:
            fvg_list.append({'start': high2, 'end': low0, 'type': 'bearish'})
    return fvg_list
