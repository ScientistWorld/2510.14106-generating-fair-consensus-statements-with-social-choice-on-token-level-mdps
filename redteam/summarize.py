"""Read only evaluator-written results and calculate the specified weighted rank.

The ranking service is external, so standardized ranks always name their cohort.
Population standard deviation is used. A constant metric gets denominator 1;
none of the nonzero-coefficient metrics is constant in the reported cohorts.
Missing metrics contribute zero, as specified by the audit protocol.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
EXPERIMENT = 'empirical_core_test'
METHOD = 'nash_welfare_policy'


def load(name):
    folder = AUDIT / name
    summary = json.loads((folder / 'run_summary.json').read_text())
    if summary.get('test_exit') != 0 or not summary.get('scores_written_by_test'):
        return None
    return json.loads((folder / 'scores_test.json').read_text())['experiments'][EXPERIMENT]['results']


def denominators(names, results, metrics):
    denominators = {}
    for metric in metrics:
        values = [results[n].get(METHOD, {}).get(metric) for n in names if results.get(n) is not None]
        values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
        if values:
            mean = sum(values)/len(values)
            std = math.sqrt(sum((x-mean)**2 for x in values)/len(values))
        else:
            std = 0.
        denominators[metric] = std if std > 1e-15 else 1.
    return denominators


def rank(values, reference, metrics, denominators, weight=1.0):
    total = 0.
    contributions = {}
    for name, definition in metrics.items():
        if name not in values or name not in reference:
            continue
        direction = 1 if definition['higher_is_better'] else -1
        contribution = weight * abs(definition['coefficient']) * direction * (values[name]-reference[name]) / denominators[name]
        contributions[name] = contribution
        total += contribution
    return total, contributions


def main():
    names = json.loads((AUDIT / 'variants.json').read_text())
    results = {}
    for name in names:
        if not (AUDIT / name / 'run_summary.json').exists():
            continue
        results[name] = load(name)
    reference_definition = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments'][EXPERIMENT]
    metrics = reference_definition['metrics']
    weight = reference_definition['weight']
    reference = results['reference'][METHOD]
    quality_cohort = [n for n in names if not n.startswith('shortcut_') and results.get(n) is not None]
    scales = denominators(quality_cohort, results, metrics)
    ranked = {}
    for name, methods in results.items():
        if methods is None:
            ranked[name] = {'valid_test_scores': False}
            continue
        values = methods.get(METHOD, {})
        score, contributions = rank(values, reference, metrics, scales, weight)
        ranked[name] = {'valid_test_scores': True, 'metrics': values,
                        'rank_vs_reference_quality_cohort': score,
                        'rank_contributions': contributions,
                        'reported_methods': list(methods)}
    seed_names = [f'reference_seed_{i}' for i in (11, 22, 33)]
    seed_spread = {}
    for metric in metrics:
        values = [ranked[n]['metrics'][metric] for n in seed_names if ranked.get(n, {}).get('valid_test_scores')]
        if values:
            seed_spread[metric] = {'min': min(values), 'max': max(values), 'spread': max(values)-min(values)}
    seed_ranks = [ranked[n]['rank_vs_reference_quality_cohort'] for n in seed_names if ranked.get(n, {}).get('valid_test_scores')]
    if seed_ranks:
        seed_spread['standardized_rank'] = {'min': min(seed_ranks), 'max': max(seed_ranks),
                                            'spread': max(seed_ranks)-min(seed_ranks)}
    # An admissible future-attempt cohort: two failing methods and one honest
    # improvement. The fixed reference is the comparison point, not an attempt.
    tradeoff_cohort = ['shortcut_constraint_tradeoff_a', 'shortcut_constraint_tradeoff_b', 'better_core_fairness']
    tradeoff = {}
    if all(results.get(n) is not None for n in tradeoff_cohort):
        tradeoff_scales = denominators(tradeoff_cohort, results, metrics)
        for name in tradeoff_cohort:
            score, contributions = rank(results[name].get(METHOD, {}), reference, metrics, tradeoff_scales, weight)
            tradeoff[name] = {'rank_vs_reference': score, 'contributions': contributions}
    else:
        tradeoff_scales = {}
    output = {'reference_source': 'redteam/reference/scores_test.json',
              'metric_definitions_source': 'scoring/reference.json',
              'formula': 'sum weight*abs(coefficient)*direction*(value-reference)/population_std; missing skipped',
              'experiment_weight': reference_definition['weight'],
              'quality_cohort': quality_cohort, 'quality_cohort_std': scales,
              'variants': ranked, 'seed_spread': seed_spread,
              'constraint_tradeoff_cohort': tradeoff_cohort,
              'constraint_tradeoff_std': tradeoff_scales, 'constraint_tradeoff_ranks': tradeoff}
    (AUDIT / 'summary.json').write_text(json.dumps(output, indent=2))
    for name, row in ranked.items():
        if row['valid_test_scores']:
            m = row['metrics']
            print(name, 'rank=', round(row['rank_vs_reference_quality_cohort'], 9),
                  'alpha=', m.get('max_coalition_improvement'),
                  'core=', m.get('core_constraint_satisfied'),
                  'min_utility=', m.get('min_agent_expected_utility'))
        else:
            print(name, 'NO SCORES WRITTEN BY SUCCESSFUL HIDDEN TEST')
    print('seed spread:', json.dumps(seed_spread))
    print('constraint tradeoff:', json.dumps(tradeoff))


if __name__ == '__main__':
    main()
