import pytest
from scripts.compare_emission_trials import summarize, KEYS


def trials():
    rows = []
    for repeat in range(2):
        for method, value in [('fixed_time', 100.), ('emission_model', 90.)]:
            rows.append(dict(seed=9000, repeat=repeat, controller=method,
                input_hash='input', protocol_hash='protocol', seconds=600, scenario='3x3grid',
                departed=300, arrived=250, teleport_events=0, collision_vehicle_events=0,
                sample_sha256=method, **dict.fromkeys(KEYS, value)))
    return rows


def test_repeat_summary_requires_matched_protocol_and_complete_pairs():
    rows = trials()
    rows[1]['protocol_hash'] = 'different'
    with pytest.raises(ValueError, match='différentes'):
        summarize(rows)
    with pytest.raises(ValueError, match='incomplètes'):
        summarize(trials()[:-1])


def test_mean_gain_does_not_hide_repetition_noise_or_fewer_arrivals():
    rows = trials()
    assert summarize(rows)['clear_four_metric_gain_in_this_sample']
    rows[1]['co2_kg'] = 105.
    rows[1]['sample_sha256'] = 'different'
    report = summarize(rows)
    assert report['metrics'][0]['paired_change_percent'] < 0
    assert not report['clear_four_metric_gain_in_this_sample']
    assert not report['exact_samples_reproducible']
    rows = trials()
    rows[1]['arrived'] = 249
    assert not summarize(rows)['clear_four_metric_gain_in_this_sample']


def test_rules_comparison_uses_rules_trials_and_labels():
    rows = trials()
    for row in rows:
        if row['controller'] == 'emission_model':
            row['controller'] = 'rules'
    report = summarize(rows, 'rules')
    assert report['clear_four_metric_gain_in_this_sample']
    assert report['metrics'][0]['rules_mean'] == 90.
    assert 'rl_mean' not in report['metrics'][0]
    with pytest.raises(ValueError, match='incomplètes'):
        summarize(rows)
