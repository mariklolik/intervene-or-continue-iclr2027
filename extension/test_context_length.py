import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_lease
import server_guard


class ContextLengthTest(unittest.TestCase):
    def test_guard_default_and_explicit_context_reach_sglang(self):
        for context in (None, 32768):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as directory:
                args = ["server_guard", "--root", directory, "--name", "context-test", "--gpu", "0", "--port", "18080", "--model-path", "/model", "--model", "actor", "--seconds", "600", "--image", "image"]
                if context is not None:
                    args.extend(["--context-length", str(context)])
                process = Mock(returncode=0)
                process.poll.return_value = 0
                with patch.object(sys, "argv", args), patch.object(server_guard.subprocess, "Popen", return_value=process) as launch, patch.object(server_guard.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="")), patch.object(server_guard.signal, "signal"), patch("builtins.print"):
                    try:
                        server_guard.main()
                    except SystemExit as error:
                        self.fail(f"Context argument rejected: {error}")
                command = launch.call_args.args[0]
                self.assertEqual(command.count("--context-length"), 1)
                self.assertEqual(command[command.index("--context-length") + 1], str(context or 16384))

    def test_lease_default_and_explicit_context_reach_guard(self):
        for context in (None, 32768):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lease = {"root": directory, "name": "context-test", "gpu": 0, "port": 18080, "model_path": "/model", "model": "actor", "seconds": 600, "image": "image", "config": str(root / "config.json"), "out": str(root / "out"), "workers": 32}
                if context is not None:
                    lease["context_length"] = context
                path = root / "lease.json"
                path.write_text(json.dumps(lease))
                process = Mock(returncode=0)
                process.poll.return_value = 0
                process.wait.return_value = 0
                with patch.object(sys, "argv", ["run_lease", str(path)]), patch.object(sys, "path", list(sys.path)), patch.object(run_lease.subprocess, "Popen", return_value=process) as launch, patch.object(run_lease, "await_model"), patch("sampling_preflight.verify_sampling"), patch("builtins.print"):
                    run_lease.main()
                command = launch.call_args_list[0].args[0]
                self.assertEqual(command.count("--context-length"), 1)
                self.assertEqual(command[command.index("--context-length") + 1], str(context or 16384))
                self.assertTrue((root / "context-test.stop").exists())


if __name__ == "__main__":
    unittest.main()
