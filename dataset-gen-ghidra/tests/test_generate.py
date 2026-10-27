"""Run with: python -m unittest discover -s dataset-gen-ghidra/tests -v."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import pickle
import signal
import subprocess
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch


# These tests exercise orchestration, not ELF parsing; no Ghidra install is needed.
source = Path(__file__).resolve().parents[1] / "generate.py"
spec = importlib.util.spec_from_file_location("generate", source)
generate = importlib.util.module_from_spec(spec)
try:
    import elftools.elf.elffile
except ImportError:
    elf_module = types.ModuleType("elftools.elf.elffile")
    elf_module.ELFFile = Mock()
    with patch.dict("sys.modules", {"elftools.elf.elffile": elf_module}):
        spec.loader.exec_module(generate)
else:
    spec.loader.exec_module(generate)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = generate.Runner.__new__(generate.Runner)
        self.runner.ghidra = "/fake/analyzeHeadless"
        self.runner.project_dir = str(self.root)
        self.runner.output_dir = str(self.root / "output")
        self.runner.env = {}
        self.runner.timeout = 10
        self.runner.verbose = False

    def initialize(self, project_dir):
        args = argparse.Namespace(
            ghidra="/fake/analyzeHeadless", binaries_dir=str(self.root),
            output_dir=str(self.root / "output"), project_dir=project_dir,
            num_files=None, verbose=False, num_threads=1, timeout=10,
        )
        with patch.object(generate.Runner, "run") as run, patch.object(generate.tf, "tempdir", None):
            runner = generate.Runner(args)
        return runner, run

    def invoke(self):
        return self.runner.run_decompiler(
            {"PREFIX": "original-binary"}, str(self.root), "/tmp/input", "/scripts/debug.py", timeout=10,
        )

    def process_with_artifacts(self, command, **kwargs):
        project_dir = Path(command[1])
        self.assertEqual(project_dir.parent, self.root)
        self.assertTrue(kwargs["start_new_session"])
        (project_dir / "project.rep").mkdir()
        (project_dir / "project.gpr").touch()
        (project_dir / "project.lock").touch()
        self.project_dir = project_dir
        return self.process

    def test_creates_custom_project_parent(self):
        parent = self.root / "new" / "projects"
        runner, run = self.initialize(str(parent))
        self.assertTrue(parent.is_dir())
        self.assertEqual(runner.project_dir, str(parent))
        run.assert_called_once()

    def test_default_project_parent_is_output(self):
        runner, _ = self.initialize(None)
        self.assertEqual(runner.project_dir, str(self.root / "output"))

    def test_invalid_project_parent_fails_before_workers(self):
        parent = self.root / "file"
        parent.touch()
        with patch.object(generate.Runner, "run") as run:
            with self.assertRaises(NotADirectoryError):
                self.initialize(str(parent))
            run.assert_not_called()

    def test_success_cleans_all_artifacts(self):
        self.process = Mock(returncode=0)
        self.process.communicate.return_value = (b"ok", b"")
        with patch.object(generate.subprocess, "Popen", side_effect=self.process_with_artifacts):
            self.assertTrue(self.invoke())
        self.assertFalse(self.project_dir.exists())

    def test_nonzero_exit_reports_binary_stage_and_output(self):
        self.process = Mock(returncode=2)
        self.process.communicate.return_value = (b"stdout", b"stderr\xff")
        errors = io.StringIO()
        with patch.object(generate.subprocess, "Popen", side_effect=self.process_with_artifacts), redirect_stderr(errors):
            self.assertFalse(self.invoke())
        for text in ("original-binary", "debug.py", "status 2", "stdout", "stderr"):
            self.assertIn(text, errors.getvalue())
        self.assertFalse(self.project_dir.exists())

    def test_timeout_waits_for_exit_before_cleanup(self):
        self.process = Mock(pid=123, returncode=-15)
        def communicate(timeout):
            self.assertTrue(self.project_dir.exists())
            if timeout == 10:
                raise subprocess.TimeoutExpired("ghidra", timeout)
            self.assertEqual(timeout, 5)
            return b"partial output", b""
        self.process.communicate.side_effect = communicate
        with patch.object(generate.subprocess, "Popen", side_effect=self.process_with_artifacts), patch.object(generate.os, "killpg") as kill, redirect_stderr(io.StringIO()) as errors:
            self.assertFalse(self.invoke())
        self.assertIn("timed out", errors.getvalue())
        self.assertIn("partial output", errors.getvalue())
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGTERM))
        self.assertFalse(self.project_dir.exists())

    def test_timeout_escalates_and_reaps_before_cleanup(self):
        self.process = Mock(pid=123, returncode=-9)
        def communicate(timeout=None):
            self.assertTrue(self.project_dir.exists())
            if timeout is not None:
                raise subprocess.TimeoutExpired("ghidra", timeout)
            return b"", b""
        self.process.communicate.side_effect = communicate
        with patch.object(generate.subprocess, "Popen", side_effect=self.process_with_artifacts), patch.object(generate.os, "killpg") as kill, redirect_stderr(io.StringIO()):
            self.assertFalse(self.invoke())
        self.assertEqual([c.args for c in kill.call_args_list], [(123, signal.SIGTERM), (123, signal.SIGKILL)])
        self.assertFalse(self.project_dir.exists())

    def test_exit_before_signal_is_harmless(self):
        process = Mock(pid=123)
        process.communicate.return_value = (b"", b"")
        with patch.object(generate.os, "killpg", side_effect=ProcessLookupError):
            self.assertEqual(self.runner.stop_decompiler(process), (b"", b""))

    def test_launch_failure_cleans_directory(self):
        with patch.object(generate.subprocess, "Popen", side_effect=FileNotFoundError), redirect_stderr(io.StringIO()):
            self.assertFalse(self.invoke())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_concurrent_invocations_have_independent_cleanup(self):
        barrier = threading.Barrier(2)
        directories = []
        unrelated = self.root / "unrelated.rep"
        unrelated.mkdir()
        def popen(command, **kwargs):
            directory = Path(command[1])
            directories.append(directory)
            (directory / "project.rep").mkdir()
            process = Mock(returncode=0)
            def communicate(**kwargs):
                barrier.wait(timeout=5)
                self.assertTrue(directory.exists())
                return b"", b""
            process.communicate.side_effect = communicate
            return process
        with patch.object(generate.subprocess, "Popen", side_effect=popen), ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.invoke(), range(2)))
        self.assertEqual(results, [True, True])
        self.assertEqual(len(set(directories)), 2)
        self.assertTrue(all(not d.exists() for d in directories))
        self.assertTrue(unrelated.exists())

    def test_sidecar_is_closed_before_launch_and_removed_on_failure(self):
        binary = str(self.root / "binary")
        def launch(*args, **kwargs):
            with open(binary + ".p", "rb") as f:
                self.assertEqual(pickle.load(f), {"var"})
            raise RuntimeError("unexpected failure")
        with patch.object(self.runner, "run_decompiler", side_effect=launch):
            with self.assertRaises(RuntimeError):
                self.runner.run_stage({}, binary, "debug.py", {"var"})
        self.assertFalse(Path(binary + ".p").exists())

    def test_failed_binary_skips_dump_and_next_binary_runs(self):
        (self.root / "bad").write_bytes(b"binary one")
        (self.root / "good").write_bytes(b"binary two")
        calls = []
        sidecars = []
        def launch(env, parent, file_path, script, **kwargs):
            sidecars.append(Path(file_path + ".p"))
            self.assertTrue(sidecars[-1].exists())
            calls.append((env["PREFIX"].split("_", 1)[1], script))
            return not env["PREFIX"].endswith("_bad")
        with patch.object(generate.subprocess, "check_call"), patch.object(generate.subprocess, "check_output"), patch.object(self.runner, "extract_dwarf_var_names", return_value={"var"}), patch.object(self.runner, "run_decompiler", side_effect=launch):
            self.runner.run_one((str(self.root), "bad"))
            self.runner.run_one((str(self.root), "good"))
        self.assertEqual(calls, [("bad", self.runner.COLLECT), ("good", self.runner.COLLECT), ("good", self.runner.DUMP_TREES)])
        self.assertTrue(all(not p.exists() for p in sidecars))


if __name__ == "__main__":
    unittest.main()
