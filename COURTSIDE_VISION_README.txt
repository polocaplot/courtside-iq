COURTSIDE VISION — experimental local video tracking

1. Open COURTSIDE IQ and navigate to COURTSIDE VISION.
2. First run: open the app's Contents/Resources/source folder and double-click INSTALL_VISION.command.
3. Supply legally obtained video via the COURTSIDE VISION upload widget or copy MP4/MOV files to ~/.courtsideiq/vision_inbox/.
4. Click Process inbox. It saves tracked video in ~/.courtsideiq/vision_outputs/ and raw detections to ~/.courtsideiq/nba_archive.sqlite3.
5. Optional: run WATCH_VISION.command for continuous folder monitoring WHILE your Mac is awake and the Terminal window remains open.

What is automatic: detect generic people and balls; maintain track IDs where possible; persist observations, visible-motion pixels/sec, proximity events, and annotated MP4.
What is NOT automatic: obtaining copyrighted NBA broadcasts; reliable player identities; court calibration; all ten players when off-screen; validated passing, shot contests, team matchups, shot success, exact speed, NBA play-by-play sync, or retraining.
No unvalidated vision data is automatically injected into predictions.

The first model download requires internet; the video processing happens locally. Processing full games can take hours on a CPU. First test with a short clip. Video tracking data can consume substantial disk space.
