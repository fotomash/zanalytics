import datetime
from analysis.wyckoff.state_machine import WyckoffStateMachine, WyckoffPhase, ZBar


def make_zbars(data):
    bars = [ZBar(ts, o, h, l, c, v) for ts, o, h, l, c, v in data]
    for b in bars:
        if getattr(b, "calculate_heuristic_delta", None):
            b.calculate_heuristic_delta()
    return bars


def test_accumulation_d_to_e_transition():
    data = [
        (datetime.datetime(2023, 1, 1, 9, 0), 105, 105.5, 104, 104.5, 1000),
        (datetime.datetime(2023, 1, 1, 9, 1), 104.5, 104.8, 102, 102.2, 1500),  # PS
        (datetime.datetime(2023, 1, 1, 9, 2), 102.2, 102.5, 100, 100.5, 3000),  # SC
        (datetime.datetime(2023, 1, 1, 9, 3), 100.5, 103.5, 100.5, 103.0, 2000),  # AR_acc
        (datetime.datetime(2023, 1, 1, 9, 4), 103.0, 103.2, 101, 101.2, 800),   # ST_Acc
        (datetime.datetime(2023, 1, 1, 9, 5), 101.2, 101.5, 99.5, 99.8, 700),   # break bar spring
        (datetime.datetime(2023, 1, 1, 9, 6), 99.8, 102.0, 99.7, 101.9, 1800),   # Spring recovery
        (datetime.datetime(2023, 1, 1, 9, 7), 101.9, 102.5, 101.5, 101.8, 600),  # Test
        (datetime.datetime(2023, 1, 1, 9, 8), 101.8, 104.2, 103.5, 104.1, 1500),  # breakout bar
        (datetime.datetime(2023, 1, 1, 9, 9), 104.1, 104.3, 103.6, 104.0, 1200),  # BU
    ]
    bars = make_zbars(data)

    sm = WyckoffStateMachine()
    sm.process_event("PS", 1, bars)
    sm.process_event("SC", 2, bars)
    sm.process_event("AR_acc", 3, bars)
    sm.process_event("ST_Acc", 4, bars)
    sm.process_event("Spring", 6, bars)
    sm.process_event("Test", 7, bars)
    sm.process_event("BU", 9, bars)

    assert sm.current_phase == WyckoffPhase.ACCUMULATION_E


def test_distribution_full_cycle():
    data = [
        (datetime.datetime(2023, 1, 2, 9, 0), 107, 108, 106, 107.5, 800),
        (datetime.datetime(2023, 1, 2, 9, 1), 108, 110, 107.5, 109.5, 1500),  # BC
        (datetime.datetime(2023, 1, 2, 9, 2), 109.5, 109.8, 105, 106, 1400),  # AR_dist
        (datetime.datetime(2023, 1, 2, 9, 3), 106, 109.5, 106, 108, 1200),  # ST_dist
        (datetime.datetime(2023, 1, 2, 9, 4), 108, 112, 107, 111, 1000),  # break bar UT
        (datetime.datetime(2023, 1, 2, 9, 5), 111, 111.2, 109, 109.5, 1000),  # UT
        (datetime.datetime(2023, 1, 2, 9, 6), 109.5, 113.5, 108, 112, 900),  # break bar UTAD
        (datetime.datetime(2023, 1, 2, 9, 7), 112, 112.5, 108.5, 108.7, 1000),  # UTAD
        (datetime.datetime(2023, 1, 2, 9, 8), 108.7, 109, 106, 106.5, 900),  # SOW
        (datetime.datetime(2023, 1, 2, 9, 9), 106.5, 106.8, 103, 104.5, 1000),  # SOW_break
    ]
    bars = make_zbars(data)

    sm = WyckoffStateMachine()
    sm.process_event("BC", 1, bars)
    assert sm.current_phase == WyckoffPhase.DISTRIBUTION_A

    sm.process_event("AR_dist", 2, bars)
    assert sm.current_phase == WyckoffPhase.DISTRIBUTION_B

    sm.process_event("ST_dist", 3, bars)
    assert sm.current_phase == WyckoffPhase.DISTRIBUTION_B

    sm.process_event("UT", 5, bars)
    assert sm.current_phase == WyckoffPhase.DISTRIBUTION_C

    sm.process_event("UTAD", 7, bars)
    assert sm.current_phase == WyckoffPhase.DISTRIBUTION_D

    sm.process_event("SOW_break", 9, bars)
    assert sm.current_phase == WyckoffPhase.DISTRIBUTION_E
