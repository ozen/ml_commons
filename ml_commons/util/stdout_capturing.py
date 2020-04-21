import os
import sys
import subprocess
from contextlib import contextmanager


@contextmanager
def capture_stdout(file_path):
    """
    Duplicate stdout and stderr to a file on the file descriptor level.
    https://github.com/IDSIA/sacred/blob/0c6267943764c57cdf14eef21163454d0322ee77/sacred/stdout_capturing.py
    http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    http://stackoverflow.com/a/651718/1388435
    http://stackoverflow.com/a/22434262/1388435
    """
    with open(file_path, mode="w+") as target:
        original_stdout_fd = 1
        original_stderr_fd = 2

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        # start_new_session=True to move process to a new process group
        # this is done to avoid receiving KeyboardInterrupts
        tee_stdout = subprocess.Popen(
            ["tee", "-a", target.name],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=1,
        )
        tee_stderr = subprocess.Popen(
            ["tee", "-a", target.name],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=2,
        )

        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)

        try:
            yield sys.stdout
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            tee_stdout.wait(timeout=1)
            tee_stderr.wait(timeout=1)

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
