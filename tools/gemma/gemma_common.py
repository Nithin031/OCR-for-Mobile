"""Shared helpers for the G0 pipeline: paths, hashing, measured subprocesses, the runner client."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
GEMMA_TOOLS = REPO_ROOT / "tools" / "gemma"
REF_GEMMA = REPO_ROOT / "tools" / "reference" / "gemma"
PIN_FILE = GEMMA_TOOLS / "llama_cpp_pin.json"
SOURCES_FILE = GEMMA_TOOLS / "gemma_sources.json"
CONFIG_FILE = REF_GEMMA / "gemma_config.json"
TASKS_FILE = REF_GEMMA / "golden_llm" / "tasks.json"
IS_WINDOWS = os.name == "nt"


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config_value(cfg: dict, section: str, key: str) -> Any:
    try:
        return cfg[section][key]["value"]
    except KeyError as e:
        raise SystemExit(f"gemma_config.json is missing {section}.{key}.value") from e


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def physical_cores() -> int:
    try:
        import psutil  # optional
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def find_binary(build_dir: Path, name: str) -> Path:
    exe = name + (".exe" if IS_WINDOWS else "")
    for cand in (build_dir / "bin" / exe, build_dir / "bin" / "Release" / exe, build_dir / "bin" / "RelWithDebInfo" / exe):
        if cand.is_file():
            return cand
    raise SystemExit(f"{exe} not found under {build_dir / 'bin'} - run the 'llama' step first")


def human_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000.0
    return str(n)


# ---------------------------------------------------------------- peak memory

class _WinPeak:
    """Tracks PeakWorkingSetSize of a child process on Windows (psapi via kernel32)."""

    def __init__(self, proc: subprocess.Popen):
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        self._ctypes = ctypes
        self._pmc_type = PMC
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._fn = k32.K32GetProcessMemoryInfo
        self._fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        self._fn.restype = wintypes.BOOL
        self._handle = wintypes.HANDLE(int(proc._handle))  # noqa: SLF001 - Popen keeps the handle open after exit
        self._proc = proc
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _query(self) -> None:
        pmc = self._pmc_type()
        pmc.cb = self._ctypes.sizeof(pmc)
        if self._fn(self._handle, self._ctypes.byref(pmc), pmc.cb):
            self.peak = max(self.peak, int(pmc.PeakWorkingSetSize))

    def _poll(self) -> None:
        while not self._stop.wait(0.25):
            self._query()

    def finish(self) -> int | None:
        self._stop.set()
        self._thread.join()
        self._query()
        return self.peak or None


class _PosixPeak:
    """Tracks a child's peak resident memory on Linux/macOS by polling while it runs.

    Not ru_maxrss: on Linux the kernel carries the forking parent's RSS across exec into the child's
    ru_maxrss, so a small tool launched from a large Python process would report the parent's size.
    Linux: VmHWM of the child's own address space. macOS: max of psutil RSS samples.
    """

    def __init__(self, proc: subprocess.Popen):
        self._pid = proc.pid
        self.peak = 0
        self._stop = threading.Event()
        self._psutil_proc = None
        if sys.platform != "linux":
            try:
                import psutil  # noqa: PLC0415
                self._psutil_proc = psutil.Process(self._pid)
            except Exception:  # noqa: BLE001 - no psutil or process already gone
                self._psutil_proc = None
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _query(self) -> None:
        try:
            if sys.platform == "linux":
                with open(f"/proc/{self._pid}/status", "rb") as f:
                    for line in f:
                        if line.startswith(b"VmHWM:"):
                            self.peak = max(self.peak, int(line.split()[1]) * 1024)
                            break
            elif self._psutil_proc is not None:
                self.peak = max(self.peak, int(self._psutil_proc.memory_info().rss))
        except (OSError, ValueError, Exception):  # noqa: BLE001 - process exiting
            pass

    def _poll(self) -> None:
        self._query()
        while not self._stop.wait(0.1):
            self._query()

    def finish(self) -> int | None:
        self._stop.set()
        self._thread.join()
        return self.peak or None


def _peak_tracker(proc: subprocess.Popen):
    return _WinPeak(proc) if IS_WINDOWS else _PosixPeak(proc)


class Measured(NamedTuple):
    returncode: int
    peak_rss_bytes: int | None
    wall_s: float


def run_measured(cmd: list, log_path: Path, cwd: Path | None = None, env: dict | None = None) -> Measured:
    """Runs cmd with stdout+stderr appended to log_path; returns exit code, peak resident memory, wall time.

    Peak memory of the child's own process: Windows PeakWorkingSetSize, Linux VmHWM, macOS sampled RSS.
    Resident pages of memory-mapped model files count on all three.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(c) for c in cmd]
    with open(log_path, "ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8"))
        log.flush()
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        tracker = _peak_tracker(proc)
        proc.wait()
        peak = tracker.finish()
        return Measured(proc.returncode, peak, time.monotonic() - t0)


# ---------------------------------------------------------------- runner client

class RunnerError(Exception):
    pass


class Runner:
    """Client for runner/gemma_runner (one JSON request/response per line)."""

    def __init__(self, binary: Path, model: Path, stderr_log: Path, *, vocab_only: bool = False,
                 n_ctx: int = 4096, n_batch: int = 2048, n_ubatch: int = 512, threads: int = 4,
                 threads_batch: int = 4, flash_attn: str = "disabled", swa_full: bool = True,
                 load_mode: str = "mmap", extra_bufts: bool = True):
        args = [str(binary), "--model", str(model), "--n-ctx", str(n_ctx), "--n-batch", str(n_batch),
                "--n-ubatch", str(n_ubatch), "--threads", str(threads), "--threads-batch", str(threads_batch),
                "--flash-attn", flash_attn, "--swa-full", "1" if swa_full else "0", "--load-mode", load_mode,
                "--kv-type", "f16", "--extra-bufts", "1" if extra_bufts else "0"]
        if vocab_only:
            args.append("--vocab-only")
        stderr_log = Path(stderr_log)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(stderr_log, "ab")
        self._log.write(("\n$ " + " ".join(args) + "\n").encode("utf-8"))
        self._log.flush()
        self.args = args
        self.log_path = stderr_log
        self._t0 = time.monotonic()
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        self._peak = _peak_tracker(self.proc)
        self.peak_rss_bytes: int | None = None
        try:
            self.ready = self._read()
        except RunnerError:
            self.close()
            raise
        if not self.ready.get("ok"):
            self.close()
            raise RunnerError(f"runner failed to start: {self.ready.get('error')} (log: {stderr_log})")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RunnerError(f"gemma_runner exited unexpectedly; see {self.log_path}")
        return json.loads(line.decode("utf-8"))

    def request(self, obj: dict) -> dict:
        self.proc.stdin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("ascii"))
        self.proc.stdin.flush()
        resp = self._read()
        if not resp.get("ok"):
            raise RunnerError(resp.get("error", "unknown runner error"))
        return resp

    def count_tokens(self, messages: list[dict]) -> int:
        return int(self.request({"op": "count", "messages": messages})["n_tokens"])

    def render(self, messages: list[dict]) -> str:
        return self.request({"op": "render", "messages": messages, "add_assistant": True})["text"]

    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
        return self.request({"op": "tokenize", "text": text, "add_special": add_special,
                             "parse_special": parse_special})["tokens"]

    def generate(self, messages: list[dict], n_predict: int) -> dict:
        return self.request({"op": "generate", "messages": messages, "n_predict": n_predict})

    def close(self) -> int | None:
        if self.proc.poll() is None and self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.write(b'{"op":"quit"}\n')
                self.proc.stdin.flush()
            except OSError:
                pass
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()
        self.peak_rss_bytes = self._peak.finish()
        if self.proc.stdout:
            self.proc.stdout.close()
        self._log.close()
        return self.peak_rss_bytes

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
