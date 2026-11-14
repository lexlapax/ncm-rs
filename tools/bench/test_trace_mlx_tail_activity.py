import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import trace_mlx_tail_activity as runner


class TraceRecorderTests(unittest.TestCase):
    def test_saving_is_not_given_a_post_target_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic"
            binary.touch()
            output = root / "output"
            observations = []

            class Recorder:
                returncode = None

                def __init__(self, _command, *, stdout, **_kwargs):
                    stdout.write(b"Starting recording\n")

                def poll(self):
                    return self.returncode

                def wait(self):
                    state = json.loads((output / "metadata.json").read_text())
                    observations.append(state["status"])
                    self.returncode = 0
                    return 0

            argv = ["trace_mlx_tail_activity.py", str(root), str(binary), str(root), str(output)]
            with patch.object(runner.os, "uname", return_value=SimpleNamespace(sysname="Darwin")), \
                    patch.object(runner.subprocess, "Popen", Recorder), \
                    patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                    patch("sys.argv", argv):
                runner.main()
            self.assertEqual(observations, ["saving"])
            self.assertEqual(json.loads((output / "metadata.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
