COURTSIDE IQ — PERSISTENT BACKGROUND COLLECTION

1. Unzip the app and launch it normally once if desired.
2. In Finder, right-click COURTSIDE IQ.app > Show Package Contents.
3. Go to Contents/Resources/source and double-click INSTALL_BACKGROUND_SYNC.command.
4. Follow the Terminal prompts. macOS may ask to allow this script.

The installer creates ~/.courtsideiq/collector_runtime with its own Python
virtual environment and registers a LaunchAgent. This location does not move
when you replace COURTSIDE IQ.app with a new download.

Every six hours WHILE YOUR MAC IS AWAKE AND LOGGED IN, it checks recent NBA
games, resumes older missing days, and retries incomplete box scores and PBP.
It records failures and does not invent data when sources are unavailable.

Database: ~/.courtsideiq/nba_archive.sqlite3
Status report: ~/.courtsideiq/recovery_report.json
Detailed logs: ~/.courtsideiq/archive_sync.log and archive_sync_errors.log

IMPORTANT:
- The initial historical catch-up can take many runs (30 older days per run).
- No video downloads or optical tracking are added by this update.
- macOS sleep, network outages, and unavailable feeds delay collection; later
  successful runs attempt to catch up.
- Re-run INSTALL_BACKGROUND_SYNC.command after an update if you want the
  stable collector to use updated collector code; existing collector continues
  working even without reinstalling.
- Before replacing the app, do NOT delete ~/.courtsideiq.
- For backups, copy the database while the app/collector is stopped, or use
  SQLite's online backup API (not just the .sqlite3 file while writes run).
