AUTOMATIC VIDEO DISCOVERY & COURTSIDE VISION — NO SEPARATE STARTUP SCRIPT

Opening COURTSIDE IQ now automatically registers a persistent, per-user video
worker. It runs at login, periodically discovers candidate video metadata
(every six hours), and checks local video inboxes roughly every 90 seconds.
The NBA play-by-play collector you already installed is left untouched.

You DO NOT need to run INSTALL_AUTOMATIC_VIDEO.command or WATCH_VISION.command.
The older video LaunchAgent is retired automatically to avoid duplicate work.

OPTIONAL INPUTS:
- For YouTube *metadata discovery only*, set your YouTube Data API key in the app.
- To analyze footage you are authorized to use, add local MP4/MOV files to
  ~/.courtsideiq/vision_inbox/; for play-specific clips use
  ~/.courtsideiq/event_clips/ named GAMEID__PLAYID.mp4.
- Optional heavyweight vision dependencies are installed in the background
  only when there are local video files to process and internet is available.

IMPORTANT: This DOES NOT download NBA/YouTube video, bypass DRM, or obtain
full-game footage automatically. YouTube metadata is not video frames.
A full play-by-play feed covers recorded events, not off-ball movement.

Status: ~/.courtsideiq/video_autopilot_status.json
Logs: ~/.courtsideiq/video_autopilot*.log
Data: ~/.courtsideiq/ (outside the replaceable app bundle)
Runs while your Mac is awake and your account is logged in.
