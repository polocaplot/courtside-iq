"""Non-blocking archive runner, concurrency-safe across app and launchd."""
from __future__ import annotations
import fcntl
from pathlib import Path
import sys
from nba_archive import cli

if __name__ == "__main__":
    root=Path.home()/".courtsideiq"
    root.mkdir(parents=True,exist_ok=True)
    with (root/"archive_sync.lock").open("w") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: sys.exit(0)
        cli()
