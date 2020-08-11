import os
import platform
import shutil
import sys
import subprocess
from io import StringIO
from wrapt import ObjectProxy


def flush():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, OSError):
        pass  # unsupported


class TeeingStreamProxy(ObjectProxy):
    """A wrapper around stdout or stderr that duplicates all output to out."""
    def __init__(self, wrapped, out):
        super().__init__(wrapped)
        self._self_out = out

    def write(self, data):
        self.__wrapped__.write(data)
        self._self_out.write(data)

    def flush(self):
        self.__wrapped__.flush()
        self._self_out.flush()


class CaptureStdout:
    def __init__(self, file_path):
        if platform.system() == "Windows":
            self.capturer = CaptureStdoutIO(file_path)
        else:
            self.capturer = CaptureStdoutFd(file_path)

    def __enter__(self):
        return self.capturer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capturer.__exit__(exc_type, exc_val, exc_tb)


class CaptureStdoutFd:
    """Duplicate stdout and stderr to a file on the file descriptor level."""

    def __init__(self, file_path):
        self.target = open(file_path, mode="w+", newline="")
        self.original_stdout_fd = 1
        self.original_stderr_fd = 2

        # Save a copy of the original stdout and stderr file descriptors
        self.saved_stdout_fd = os.dup(self.original_stdout_fd)
        self.saved_stderr_fd = os.dup(self.original_stderr_fd)

        # start_new_session=True to move process to a new process group
        # this is done to avoid receiving KeyboardInterrupts
        self.tee_stdout = subprocess.Popen(
            ["tee", "-a", self.target.name],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=1,
        )
        self.tee_stderr = subprocess.Popen(
            ["tee", "-a", self.target.name],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=2,
        )

        flush()
        os.dup2(self.tee_stdout.stdin.fileno(), self.original_stdout_fd)
        os.dup2(self.tee_stderr.stdin.fileno(), self.original_stderr_fd)

    def __enter__(self):
        return self.target

    def __exit__(self, exc_type, exc_val, exc_tb):
        flush()

        # then redirect stdout back to the saved fd
        self.tee_stdout.stdin.close()
        self.tee_stderr.stdin.close()

        # restore original fds
        os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
        os.dup2(self.saved_stderr_fd, self.original_stderr_fd)

        self.tee_stdout.wait(timeout=1)
        self.tee_stderr.wait(timeout=1)

        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)

        self.target.flush()
        self.target.close()


class CaptureStdoutIO:
    """Duplicate sys.stdout and sys.stderr to new StringIO."""

    def __init__(self, file_path):
        self.target = open(file_path, mode="w+", newline="", buffering=1)
        self.orig_stdout, self.orig_stderr = sys.stdout, sys.stderr
        flush()
        sys.stdout = TeeingStreamProxy(sys.stdout, self.target)
        sys.stderr = TeeingStreamProxy(sys.stderr, self.target)

    def __enter__(self):
        return self.target

    def __exit__(self, exc_type, exc_val, exc_tb):
        flush()
        self.target.flush()
        self.target.close()
        sys.stdout, sys.stderr = self.orig_stdout, self.orig_stderr
