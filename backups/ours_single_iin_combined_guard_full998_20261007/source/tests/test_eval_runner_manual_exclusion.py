import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "run_eval_skip_cuda_crash.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "ieve_run_eval_skip_cuda_crash", RUNNER_PATH
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


class ManualExclusionTests(unittest.TestCase):
    def test_exclusion_spec_uses_episode_list_parser(self):
        self.assertEqual(runner.parse_episode_spec("1,3-5,3"), [1, 3, 4, 5])

    def test_registration_migrates_stale_failures(self):
        summary = {
            "failed_other": [
                {"episode": 709, "returncode": -9},
                {"episode": 709, "returncode": -15},
                {"episode": 713, "returncode": 1},
            ]
        }

        added, removed = runner.register_manual_exclusions(
            summary, [709], "pathological host-memory growth / repeated SIGKILL"
        )

        self.assertEqual((added, removed), (1, 2))
        self.assertEqual(
            summary["skipped_manual"],
            [{
                "episode": 709,
                "reason": (
                    "manual exclusion: pathological host-memory growth / "
                    "repeated SIGKILL"
                ),
            }],
        )
        self.assertEqual(summary["failed_other"], [{"episode": 713, "returncode": 1}])

    def _run_dry(self, extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "run_eval_skip_cuda_crash.py",
                "--episodes", "709,713",
                "--log-dir", temp_dir,
                "--dry-run",
            ] + extra_args + ["--", "/bin/echo"]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                returncode = runner.main()
            return returncode, stdout.getvalue(), Path(temp_dir)

    def test_dry_run_skips_excluded_episode(self):
        returncode, output, _ = self._run_dry([
            "--resume", "--exclude-episodes", "709"
        ])

        self.assertEqual(returncode, 0)
        self.assertIn("[SKIP_EXCLUDED] habitat_episode_id=709", output)
        self.assertIn("[RUN] habitat_episode_id=713", output)
        self.assertNotIn("[RUN] habitat_episode_id=709", output)

    def test_without_exclusions_runs_every_requested_episode(self):
        returncode, output, _ = self._run_dry([])

        self.assertEqual(returncode, 0)
        self.assertIn("[RUN] habitat_episode_id=709", output)
        self.assertIn("[RUN] habitat_episode_id=713", output)
        self.assertNotIn("[SKIP_EXCLUDED]", output)

    def test_rejects_exclusion_outside_requested_list(self):
        argv = [
            "run_eval_skip_cuda_crash.py",
            "--episodes", "709,713",
            "--exclude-episodes", "714",
            "--log-dir", "/tmp/unused",
            "--dry-run",
            "--", "/bin/echo",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            runner.parse_args()


if __name__ == "__main__":
    unittest.main()
