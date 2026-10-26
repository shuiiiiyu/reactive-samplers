import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_coco10_grid import report


class CocoReportTests(unittest.TestCase):
    def test_pairs_keep_real_outputs_timesteps_and_recovery_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for model in ('sd3', 'sdxl'):
                for arm in ('default', 'skip'):
                    folder = root / f'{model}_{arm}'
                    trace = folder / 'trajectories/p/events.jsonl'
                    trace.parent.mkdir(parents=True)
                    skip = arm == 'skip'
                    row = {'id': 'p', 'prompt': 'fixture', 'seed': 7, 'steps': 2 if skip else 4,
                        'nfe': 2 if skip else 4, 'scores': {'hps_v2_1': .3, 'clip_score': 30.},
                        'end_to_end_seconds': 8. if skip else 2., 'denoiser_seconds': 1. if skip else 2.,
                        'agent_seconds': 5. if skip else 0., 'agent_calls': 1 if skip else 0,
                        'valid_for_benchmark': not (skip and model == 'sdxl'),
                        'timesteps': [1000, 500] if skip else [1000, 750, 500, 250],
                        'kept_indices': [0, 2], 'skipped_indices': [1, 3]}
                    (folder / 'results.json').write_text(json.dumps({'images': [row]}))
                    event = {'event': 'step', 'timestep': 1000, 'grid_decisions': [{'index': 1, 'skip': skip}]}
                    if skip:
                        answer = folder / 'decisions/00001/answer.txt'
                        answer.parent.mkdir(parents=True)
                        answer.write_text('Observed stable features; try skipping one node.')
                        event['agent_decision'] = 1
                    trace.write_text(json.dumps(event) + '\n')
            report(root)
            with (root / 'comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 2)
            self.assertEqual(float(pairs[0]['wall_speedup']), .25)
            self.assertEqual(float(pairs[0]['nfe_reduction']), .5)
            self.assertEqual(pairs[1]['valid_pair'], 'False')
            traces = json.loads((root / 'decision_trace.json').read_text())
            self.assertIn('Observed stable features', next(r['agent_output_text'] for r in traces if r['arm'] == 'skip'))
            self.assertIn('无失败/回退的配对 0/10', (root / 'comparison.md').read_text())
