import ctypes
import errno
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.char.common import atomic_file, write_json


@contextmanager
def windows_reader_without_delete_sharing(path):
    """An editor/scanner-style handle: reads/writes allowed, replacing denied."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    mutex = threading.Lock()

    def release():
        nonlocal handle
        with mutex:
            if handle is not None:
                kernel.CloseHandle(handle)
                handle = None

    try:
        yield release
    finally:
        release()


class CharacterFileIOTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_progress_locked_by_reader_does_not_abort_and_next_update_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            write_json(target, {"epoch": 14})
            with windows_reader_without_delete_sharing(target):
                self.assertFalse(write_json(target, {"epoch": 15}, required=False))
                self.assertEqual(json.loads(target.read_text()), {"epoch": 14})
            self.assertTrue(write_json(target, {"epoch": 15}, required=False))
            self.assertEqual(json.loads(target.read_text()), {"epoch": 15})
            self.assertEqual(list(Path(directory).iterdir()), [target])

    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_required_write_retries_until_real_windows_reader_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "training_state.json"
            write_json(target, {"epoch": 14})
            with windows_reader_without_delete_sharing(target) as release:
                timer = threading.Timer(0.12, release)
                timer.start()
                try:
                    self.assertTrue(write_json(target, {"epoch": 15}))
                finally:
                    timer.join(timeout=1)
            self.assertEqual(json.loads(target.read_text()), {"epoch": 15})

    def test_failed_critical_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "last.keras"
            target.write_bytes(b"previous checkpoint")
            denied = PermissionError(errno.EACCES, "locked")
            with patch("src.char.common.os.replace", side_effect=denied), patch("src.char.common.time.sleep"):
                with self.assertRaises(PermissionError):
                    with atomic_file(target) as staging:
                        self.assertEqual(staging.suffix, ".keras")
                        staging.write_bytes(b"new checkpoint")
            self.assertEqual(target.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_optional_progress_does_not_hide_disk_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            with patch("src.char.common.os.replace", side_effect=OSError(errno.ENOSPC, "disk full")):
                with self.assertRaises(OSError):
                    write_json(target, {"epoch": 15}, required=False)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_concurrent_writers_have_unique_staging_files_and_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            barrier = threading.Barrier(4)

            def writer(index):
                with atomic_file(target) as staging:
                    staging.write_text(json.dumps({"writer": index, "payload": [index] * 100}))
                    barrier.wait(timeout=5)
                    return staging.name

            with ThreadPoolExecutor(max_workers=4) as pool:
                names = list(pool.map(writer, range(4)))
            self.assertEqual(len(set(names)), 4)
            final = json.loads(target.read_text())
            self.assertEqual(final["payload"], [final["writer"]] * 100)
            self.assertEqual(list(Path(directory).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
