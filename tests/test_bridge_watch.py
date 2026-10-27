"""A lost SSH connection must not permanently stop a queued experiment's bridge."""
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch


class BridgeWatchTests(unittest.TestCase):
    def test_transport_retry_resumes_polling_and_preserves_interrupt(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/runpod_codex_bridge.py'
        for error in (subprocess.CalledProcessError(255, ['ssh']),
                      subprocess.TimeoutExpired(['ssh'], 60)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cli = root / 'codex'
                cli.touch()
                argv = [str(script), '--run-dir', '/project/sampler_runs/test',
                        '--project-root', '/project', '--codex', str(cli),
                        '--codex-home', str(root / 'auth'), '--reasoning-effort', 'low']
                connected = subprocess.CompletedProcess(['ssh'], 0, '', '')
                with patch.object(sys, 'argv', argv), patch('subprocess.run',
                        side_effect=[error, connected, KeyboardInterrupt]) as run, \
                        patch('time.sleep') as sleep, self.assertRaises(KeyboardInterrupt):
                    runpy.run_path(str(script), run_name='__main__')
                self.assertEqual(run.call_count, 3)
                self.assertEqual(sleep.call_args_list, [call(5), call(1)])


if __name__ == '__main__':
    unittest.main()
