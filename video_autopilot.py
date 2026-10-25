"""One persistent video-discovery/authorized-local-video worker.

No broadcast downloads, DRM workarounds, or unverified optical claims.
Started by a per-user LaunchAgent installed automatically when the app opens.
"""
from __future__ import annotations
import fcntl, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path.home()/'.courtsideiq'
STATE=ROOT/'video_autopilot_status.json'
LOCK=ROOT/'video_autopilot.lock'
INTERVAL=90
DISCOVERY_INTERVAL=6*3600

def write_status(**fields):
    ROOT.mkdir(parents=True,exist_ok=True)
    payload={}
    try:payload=json.loads(STATE.read_text())
    except (OSError,ValueError):pass
    payload.update(fields)
    payload['updated_utc']=datetime.now(timezone.utc).isoformat()
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2,default=str))
    os.replace(tmp,STATE)

def has_local_clips():
    from courtside_vision import INBOX,SUPPORTED
    from automatic_video import CLIPS
    return any(p.is_file() and p.suffix.lower() in SUPPORTED for folder in (INBOX,CLIPS) if folder.is_dir() for p in folder.iterdir())

def ensure_vision_dependencies():
    try:
        import cv2, ultralytics
        return True
    except ImportError:
        pass
    # Only download optional heavyweight packages if user has provided video.
    if not has_local_clips():return False
    write_status(vision_dependencies='installing')
    try:
        subprocess.run([sys.executable,'-m','pip','install','ultralytics>=8.3,<9','opencv-python-headless>=4.8,<5'],
                       check=True,timeout=1200,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        import cv2, ultralytics
        write_status(vision_dependencies='ready')
        return True
    except Exception as exc:
        write_status(vision_dependencies='installation_failed',vision_error=str(exc)[:500])
        return False

def process_inbox():
    from courtside_vision import INBOX,SUPPORTED,process_video
    INBOX.mkdir(parents=True,exist_ok=True)
    processed=0;errors=[]
    for path in sorted(INBOX.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:continue
        if time.time()-path.stat().st_mtime<45:continue
        try:
            result=process_video(path,max_frames=600,stride=3)
            if result.get('status')!='already_processed':processed+=1
        except Exception as exc:errors.append({'file':path.name,'error':str(exc)[:200]})
    return {'processed':processed,'errors':errors}

def tick(do_discovery=True):
    from automatic_video import run
    from nba_archive import connect
    from automatic_video import discover, process_authorized_clips
    ROOT.mkdir(parents=True,exist_ok=True)
    result={'discovery':'not_due','vision':'no_local_video'}
    ready=ensure_vision_dependencies() if has_local_clips() else False
    with connect() as db:
        if do_discovery:
            key_file=ROOT/'youtube_api_key'
            key=os.environ.get('YOUTUBE_DATA_API_KEY') or (key_file.read_text().strip() if key_file.is_file() else '')
            try:result['discovery']=discover(db,key)
            except Exception as exc:result['discovery']={'status':'error','error':str(exc)[:300]}
        if ready:
            try:result['event_clips']=process_authorized_clips(db)
            except Exception as exc:result['event_clips']={'status':'error','error':str(exc)[:300]}
    if ready:result['vision']=process_inbox()
    elif has_local_clips():result['vision']='dependencies_unavailable'
    write_status(last_run=result)
    return result

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        write_status(worker='running',pid=os.getpid())
        last_discovery=0
        while True:
            now=time.monotonic()
            due=now-last_discovery>=DISCOVERY_INTERVAL
            try:
                tick(do_discovery=due)
                if due:last_discovery=now
            except Exception as exc:
                write_status(worker='running',last_error=f'{type(exc).__name__}: {exc}',trace=traceback.format_exc()[-1500:])
            time.sleep(INTERVAL)

if __name__=='__main__':sys.exit(main())
