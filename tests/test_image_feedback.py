"""Dual scoring and delayed feedback delivery; no downloads, GPU or paid Agent calls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reactive_samplers.agent import AgentSession
from reactive_samplers.context import Context, METHODS, Prediction
from reactive_samplers.evaluation import QualityMetrics, paired_feedback
from reactive_samplers.programs import Programs
from reactive_samplers.report import write_json


class ImageFeedbackTests(unittest.TestCase):
    def test_official_scorer_contract_and_clip_calculation(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, config, clip_checkpoint = [root / name for name in ('ImageReward.pt', 'med_config.json', 'clip.pt')]
            for path in (checkpoint, config, clip_checkpoint):
                path.touch()
            image_path = root / 'image.png'
            Image.new('RGB', (8, 8)).save(image_path)
            reward = Mock()
            reward.score.return_value = .75
            reward_module = SimpleNamespace(load=Mock(return_value=reward))
            model = Mock()
            model.parameters.side_effect = lambda: iter([torch.zeros(1)])
            model.encode_image.return_value = torch.tensor([[1., 0.]])
            model.encode_text.return_value = torch.tensor([[.6, .8]])
            clip_module = SimpleNamespace(
                load=Mock(return_value=(model, lambda im: torch.zeros(3, 8, 8))),
                tokenize=lambda prompts, truncate: torch.zeros(1, 2, dtype=torch.long))
            with patch.dict('sys.modules', {'ImageReward': reward_module, 'clip': clip_module}), \
                    patch('torch.cuda.is_available', return_value=False):
                metrics = QualityMetrics(['clip_score', 'image_reward'], clip_checkpoint, checkpoint, config)
                for _ in range(2):
                    scores = metrics.score(image_path, 'fixture')
                    self.assertAlmostEqual(scores['clip_score'], 60., places=4)
                    self.assertEqual(scores['image_reward'], .75)
                reward_module.load.assert_called_once_with(str(checkpoint), device='cpu', med_config=str(config))
                self.assertEqual(reward.score.call_count, 2)
            with self.assertRaises(ValueError):
                QualityMetrics(['hps_v2_1'])

    def test_paired_scores_and_next_agent_request_for_all_methods(self):
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                programs = Programs.create(root / 'workspace', root / 'programs', method=method)
                reference = {'prompt': 'fixture', 'seed': 7, 'image': 'reference.png', 'nfe': 28,
                             'scores': {'clip_score': 30., 'image_reward': 1.}}
                result = {'id': 'image_1', 'prompt': 'fixture', 'seed': 7, 'image': 'image_1.png', 'nfe': 14,
                          'scores': {'clip_score': 29., 'image_reward': .8}}
                result.update(paired_feedback(result, [{**reference, 'seed': 8}, reference]))
                self.assertEqual(result['score_deltas']['clip_score'], -1.)
                self.assertAlmostEqual(result['score_deltas']['image_reward'], -.2)
                self.assertEqual(result['nfe_reduction'], .5)
                self.assertEqual(paired_feedback(result, [{**reference, 'seed': 8}]), {})
                # Legacy references lack ImageReward: report only known deltas.
                legacy = {**reference, 'scores': {'clip_score': 30.}}
                self.assertNotIn('image_reward', paired_feedback(result, [legacy])['score_deltas'])
                requests = []
                def backend(path, workspace):
                    requests.append(json.loads(path.read_text()))
                    return 'keep'
                agent = AgentSession(backend, root, lambda ctx, path: path.write_text('fixture'), method=method)
                ctx = Context('next image', 9, 0, 1, 28, 1., 0., 1000., (1., 0.), 0., Prediction(None), [], method=method)
                write_json(root / 'feedback.json', {'completed_images': []})
                agent.check(ctx, {}, programs)
                write_json(root / 'feedback.json', {'completed_images': [result]})
                agent.check(ctx, {}, programs)
                agent.check(ctx, {}, programs)
                self.assertEqual(requests[0]['new_completed_images'], [])
                self.assertEqual(requests[1]['new_completed_images'], [result])
                self.assertEqual(requests[1]['completed_image_count'], 1)
                self.assertEqual(requests[2]['new_completed_images'], [])
                self.assertIn('scores.image_reward', agent.instructions.read_text())


if __name__ == '__main__':
    unittest.main()
