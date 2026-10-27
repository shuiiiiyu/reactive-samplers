"""Current-node semantics, reference timing, English reports and gallery integration."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from test_cache_methods import adapter_fixture, programs_for
from reactive_samplers.agent import AgentSession
from reactive_samplers.context import METHODS
from reactive_samplers.decision_records import HEADINGS, parse_report
from reactive_samplers.gallery import Gallery
from reactive_samplers.observations import definitions, relative_change
from reactive_samplers.recovery import clone_state
from reactive_samplers.report import Recorder, write_json
from reactive_samplers.schedule import CheckSchedule
from reactive_samplers.wrapper import sample


class ObservationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_references_reports_and_gallery_for_every_method(self):
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                project = Path(temporary)
                root = project / 'sampler_runs' / method
                root.mkdir(parents=True)
                adapter, network = adapter_fixture('sd3', cfg=False)
                programs = programs_for(root, method, 'ctx.step in (1, 2)')
                contexts, requests = [], []
                report = '\n'.join(heading + ': None' for heading in HEADINGS.values())
                def backend(path, workspace):
                    request = json.loads(path.read_text())
                    requests.append(request)
                    self.assertFalse(request['decision_context']['prediction_available'])
                    self.assertEqual(request['decision_context']['scope'], 'current_node_only')
                    self.assertEqual(set(request['observation']), set(request['observation_definitions']['fields']))
                    return report
                def save(ctx, path):
                    contexts.append(clone_state(ctx))
                    path.write_text('fixture')
                agent = AgentSession(backend, root, save, method=method)
                recorder = Recorder(root / 'trajectories' / 'p', run_root=root)
                image, stats, history = sample(adapter, programs, 'fixture', 7, 5, 5,
                    CheckSchedule(), recorder, agent, method=method, max_consecutive_skips=4)
                self.assertEqual(len(contexts), 5)
                self.assertEqual([ctx.prediction.guided for ctx in contexts], [None]*5)
                expected_refs = [None, 0, 0, 0, 3] if method == 'grid_skip' else [None, 0, 1, 2, 3]
                self.assertEqual([ctx.observation_reference_step for ctx in contexts], expected_refs)
                for k, (ctx, request) in enumerate(zip(contexts, requests)):
                    reference = expected_refs[k]
                    expected = None if reference is None else relative_change(ctx.latent, contexts[reference].latent)
                    self.assertEqual(request['observation']['latent_rel_change'], expected)
                    self.assertIn('has_reference', request['observation']) if method != 'grid_skip' else self.assertNotIn('has_reference', request['observation'])
                if method == 'grid_skip':
                    torch.testing.assert_close(contexts[1].latent, contexts[2].latent, rtol=0, atol=0)
                    torch.testing.assert_close(contexts[2].latent, contexts[3].latent, rtol=0, atol=0)
                    self.assertEqual(stats['scheduler_updates'], 3)
                    self.assertEqual(stats['nfe'], 3)
                    self.assertEqual(stats['skipped_indices'], [1, 2])
                else:
                    self.assertEqual(stats['scheduler_updates'], 5)
                    self.assertNotEqual(requests[3]['observation']['since_real_delta_sigma'], 0)
                # No invented reports: incomplete output is explicitly marked.
                self.assertFalse(all(parse_report('keep').values()))
                archived = json.loads((root / 'agent_decisions/00002.json').read_text())
                self.assertTrue(archived['report_complete'])
                self.assertEqual(archived['report']['rule_changes'], 'None')
                self.assertEqual(archived['outcome']['outcome'], 'kept')
                self.assertTrue(archived['executions'])
                self.assertTrue((root / 'agent_decisions/00002.md').exists())
                (root / 'images').mkdir()
                Image.new('RGB', (8, 8)).save(root / 'images/p.png')
                write_json(root / 'results.json', {'images': [{'id': 'p', 'image': 'images/p.png',
                    'prompt': 'fixture', 'seed': 7, 'scores': {'clip_score': 30., 'image_reward': .7}}]})
                rows = Gallery(project).images()
                self.assertEqual(len(rows[0]['config']['agent_decisions']), 5)
                self.assertEqual(rows[0]['config']['agent_decisions'][1]['report']['signals_used'], 'None')

    def test_grid_skip_matches_manual_omitted_intervals_both_models(self):
        for model in ('sd3', 'sdxl'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                adapter, _ = adapter_fixture(model)
                expected, sigmas = adapter.start('fixture', 7, 5)
                for k, (sigma, target) in enumerate(zip(sigmas, sigmas[1:])):
                    if k not in (1, 2):
                        prediction = adapter.predict(expected, sigma, [])
                        expected = adapter.advance(expected, prediction, sigma, target)
                actual, stats, _ = sample(adapter, programs_for(root, 'grid_skip', 'ctx.step in (1, 2)'),
                    'fixture', 7, 5, 5, CheckSchedule(), Recorder(root / 'image'), method='grid_skip')
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(stats['nfe'], 3)
                self.assertEqual(adapter.denoiser_evaluations, 3)

    def test_agent_defined_signal_definitions_follow_analysis(self):
        from types import SimpleNamespace
        ctx = SimpleNamespace(method='prediction_cache', observation_reference_step=2, observation_reference_sigma=.5)
        custom = {'description': 'Squared sigma.', 'mathematical_definition': 'sigma[k]^2',
                  'reference': 'current node', 'range_and_units': 'squared sigma', 'missing_value': 'never null'}
        spec = definitions(ctx, SimpleNamespace(OBSERVATION_SPEC={'sigma_squared': custom}), {'sigma_squared': .25})
        self.assertEqual(spec['fields']['sigma_squared'], custom)
