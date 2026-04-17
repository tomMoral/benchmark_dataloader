"""Shared cache-control helpers using posix_fadvise."""

import ctypes
import os


# POSIX_FADV constants
POSIX_FADV_WILLNEED = 3
POSIX_FADV_DONTNEED = 6

_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return _libc


def _fadvise_files(file_paths, advice):
    """Call posix_fadvise on each file path.

    Parameters
    ----------
    file_paths : list of str or Path
        Files to advise.
    advice : int
        POSIX_FADV_DONTNEED (6) or POSIX_FADV_WILLNEED (3).
    """
    libc = _get_libc()
    for fp in file_paths:
        try:
            fd = os.open(str(fp), os.O_RDONLY)
            try:
                libc.posix_fadvise(fd, 0, 0, advice)
            finally:
                os.close(fd)
        except OSError:
            # Lustre or other FS may reject the call — silently ignore.
            pass


def evict_cache(file_paths):
    """Evict files from OS page cache (cold start)."""
    _fadvise_files(file_paths, POSIX_FADV_DONTNEED)


def readahead_cache(file_paths):
    """Request async kernel readahead (warm hint)."""
    _fadvise_files(file_paths, POSIX_FADV_WILLNEED)
