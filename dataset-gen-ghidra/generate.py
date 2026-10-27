# Specialized version for ghidra compatibility

import argparse
import subprocess
import os
import signal
import sys
import errno
import hashlib
import tempfile as tf
import pickle


from tqdm import tqdm
from multiprocessing import Pool
from typing import Iterable, Tuple, Optional

from elftools.elf.elffile import ELFFile

class Runner(object):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    COLLECT = os.path.join(file_dir, "decompiler", "debug.py")
    DUMP_TREES = os.path.join(file_dir, "decompiler", "dump_trees.py")
    
    def __init__(self, args: argparse.Namespace):
        self.ghidra = args.ghidra
        self.binaries_dir = args.binaries_dir
        self.output_dir = args.output_dir
        self._num_files = args.num_files
        self.verbose = args.verbose
        self.num_threads = args.num_threads
        self.timeout = args.timeout
        self.project_dir = os.path.abspath(args.project_dir or self.output_dir)

        self.env = os.environ.copy()

        self.env["IDALOG"] = "/dev/stdout"
        self.env["OUTPUT_DIR"] = self.output_dir

        self.make_dir(self.project_dir)
        self.make_dir(self.output_dir)
        self.make_dir(os.path.join(self.output_dir, "types"))
        self.make_dir(os.path.join(self.output_dir, "bins"))

        # Use RAM-backed memory for tmp if available
        if os.path.exists("/dev/shm"):
            tf.tempdir = "/dev/shm"
        try:
            self.run()
        except KeyboardInterrupt:
            pass

    @property
    def binaries(self) -> Iterable[Tuple[str, str]]:
        """Readable 64-bit ELFs in the binaries_dir and their paths"""

        def is_elf64(root: str, path: str) -> bool:
            file_path = os.path.join(root, path)
            try:
                with open(file_path, "rb") as f:
                    header = f.read(5)
                    # '\x7fELF' means it's an ELF file, '\x02' means 64-bit
                    return header == b"\x7fELF\x02"
            except IOError:
                return False

        return (
            (root, f)
            for root, _, files in os.walk(self.binaries_dir)
            for f in files
            if is_elf64(root, f)
        )

    @property
    def num_files(self) -> int:
        """The number of files in the binaries directory. Note that this is not
        the total number of binaries because it does not check file headers. The
        number of binary files could be lower."""
        if self._num_files is None:
            self._num_files = 0
            for _, _, files in os.walk(self.binaries_dir):
                self._num_files += len(files)
        return self._num_files

    @staticmethod
    def make_dir(dir_path):
        """Make a directory, with clean error messages."""

        try:
            os.makedirs(dir_path)
        except OSError as e:
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(f"'{dir_path}' is not a directory")
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def stop_decompiler(process):
        """Stop the process group and drain pipes before project cleanup."""
        def signal_group(sig):
            try:
                # start_new_session=True makes the child PID its process group ID.
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        signal_group(signal.SIGTERM)
        try:
            output = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            signal_group(signal.SIGKILL)
            return process.communicate()
        # A descendant can outlive the leader after closing its output pipes.
        signal_group(signal.SIGKILL)
        return output

    def run_decompiler(self, env, project_parent_dir, file_path, script, timeout=None):
        """Run a stage in an isolated project directory; return whether it succeeded."""
        script_dir, script_name = os.path.split(script)
        binary = env.get("PREFIX", file_path)
        with tf.TemporaryDirectory(prefix="ghidra-", dir=project_parent_dir) as project_dir:
            ghidracall = [
                self.ghidra, project_dir, "project", '-import', file_path,
                '-postScript', script_name, file_path + ".p", "-scriptPath", script_dir,
                '-max-cpu', "1", '-deleteProject',
            ]
            if self.verbose:
                print(f"Running {ghidracall}")
            try:
                process = subprocess.Popen(
                    ghidracall, env=env, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            except OSError as e:
                print(f"Ghidra {script_name} failed for {binary}: {e}; skipping binary",
                      file=sys.stderr)
                return False

            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stdout, stderr = self.stop_decompiler(process)
            except (KeyboardInterrupt, SystemExit):
                self.stop_decompiler(process)
                raise

            success = not timed_out and process.returncode == 0
            if not success:
                reason = (
                    f"timed out after {timeout}s" if timed_out
                    else f"exited with status {process.returncode}"
                )
                print(f"Ghidra {script_name} {reason} for {binary}; skipping binary",
                      file=sys.stderr)
            if self.verbose or not success:
                stream = sys.stdout if success else sys.stderr
                for output in (stdout, stderr):
                    if output:
                        print(output.decode(errors="replace"), file=stream)
            return success

    def run_stage(self, env, file_path, script, variables):
        """Write the stage's sidecar and remove it even if execution fails."""
        pickle_file = file_path + ".p"
        try:
            with open(pickle_file, "wb") as f:
                pickle.dump(variables, f)
            return self.run_decompiler(
                env, self.project_dir, file_path, script, timeout=self.timeout,
            )
        finally:
            if os.path.exists(pickle_file):
                os.remove(pickle_file)

    def extract_dwarf_var_names(self, filepath:str) -> Optional[set]:
        """
        Can't figure out how to extract debugging information in GHIDRA's interface, 
        so this function extracts the debugging variable/parameters names to pass into
        each ghidra call through a pickle file
        """
        variable_names = set()
        with open(filepath, 'rb') as f:
            elffile = ELFFile(f)
            if not elffile.has_dwarf_info():
                if self.verbose:
                    print(f"No dwarf info in {filepath}")
                return None
            
            # for some reason, this is throwing an exception, give it if it does so
            try:
                dwarfinfo = elffile.get_dwarf_info()
            except:
                print(f"Error extracting dwarf info from {filepath}")
                return None

            for CU in dwarfinfo.iter_CUs():
                for DIE in CU.iter_DIEs():
                    if DIE.tag != "DW_TAG_variable" and DIE.tag != "DW_TAG_formal_parameter":
                        continue
                    
                    for attr in DIE.attributes.values():
                        if attr.name == "DW_AT_name":
                            variable_names.add(attr.value.decode())
            if self.verbose:
                print(f"Extracted variable names: {variable_names}")
            return variable_names

    def run_one(self, args: Tuple[str, str]) -> None:
        path, binary = args
        new_env = self.env.copy()
        with tf.TemporaryDirectory() as tempdir:
            with tf.NamedTemporaryFile(dir=tempdir) as functions, tf.NamedTemporaryFile(
                dir=tempdir
            ) as orig, tf.NamedTemporaryFile(dir=tempdir) as stripped:
                file_path = os.path.join(path, binary)
                new_env["FUNCTIONS"] = functions.name
                # Build up hash string in 4k blocks
                file_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        file_hash.update(byte_block)
                prefix = f"{file_hash.hexdigest()}_{binary}"
                new_env["PREFIX"] = prefix
                # Try stripping first, if it fails return
                subprocess.check_call(["cp", file_path, stripped.name])
                try:
                    subprocess.check_call(["strip", stripped.name])
                except subprocess.CalledProcessError:
                    if self.verbose:
                        print(f"Could not strip {prefix}, skipping.")
                    return
                if os.path.exists(
                    os.path.join(self.output_dir, "bins", prefix + ".jsonl.gz")
                ):
                    if self.verbose:
                        print(f"{prefix} already collected, skipping")
                    return
                if os.path.exists(
                    os.path.join(self.output_dir, "types", prefix + ".jsonl.gz")
                ):
                    if self.verbose:
                        print(f"{prefix} types already collected, skipping")
                else:
                    # Collect from original
                    if self.verbose:
                        print(f"Collecting debug information")
                    subprocess.check_output(["cp", file_path, orig.name])

                    var_set = self.extract_dwarf_var_names(orig.name)
                    if var_set:
                        if not self.run_stage(new_env, orig.name, self.COLLECT, var_set):
                            return
                    else:
                        if self.verbose:
                            print(f"Unable to collect debug information for {prefix}")
                # Dump trees
                if self.verbose:
                    print(f"Dumping trees")
                self.run_stage(new_env, stripped.name, self.DUMP_TREES, set())

    def run(self):
        # File counts for progress output

        with Pool(self.num_threads) as pool:
            for p in tqdm(
                    pool.imap_unordered(self.run_one, self.binaries),
                    total=self.num_files,
                    leave=True,
                    dynamic_ncols=True,
                    unit="bin",
                    smoothing=0.1,
            ):
                pass
    pass

def main():
    parser = argparse.ArgumentParser(description="Run ghidra decompiler to generate a corpus")
    parser.add_argument(
        "--ghidra",
        metavar="GHIDRA",
        help='location of the analyzeHeadless ghidra binary',
        default='~/ghidra_10.1.4_PUBLIC/support/analyzeHeadless'
    )
    parser.add_argument(
        '-t',
        '--num-threads',
        metavar='NUM_THREADS',
        help='number of threads to use',
        default=28,
        type=int
    )
    parser.add_argument(
        '-n',
        '--num-files',
        metavar='NUM_FILES',
        help='number of binary files, None for all files',
        default=None,
        type=int
    )
    parser.add_argument(
        '-b',
        '--binaries-dir',
        metavar='BINARIES_DIR',
        help='directory containing binaries',
        required=True
    )
    parser.add_argument(
        "--timeout",
        metavar="TIMEOUT",
        help="timeout for each binary",
        default=30*60,
        type=int
    )
    parser.add_argument(
        "-o", "--output_dir", metavar="OUTPUT_DIR", help="output directory", required=True,
    )
    parser.add_argument(
        "--project-dir",
        metavar="PROJECT_DIR",
        help="parent directory where Ghidra projects are created (defaults to OUTPUT_DIR)",
        default=None,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    Runner(args)


if __name__ == '__main__':
    main()
