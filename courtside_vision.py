"""COURTSIDE VISION: local video observation pipeline.

Video must be supplied lawfully. Uses a generic pretrained object detector;
not NBA-calibrated, not official player IDs, no invented optical tracking.
Optional dependency: pip install ultralytics opencv-python-headless.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

HOME=Path.home()/'.courtsideiq'
DB=HOME/'nba_archive.sqlite3'
INBOX=HOME/'vision_inbox'
OUT=HOME/'vision_outputs'
SUPPORTED={'.mp4','.mov','.m4v','.avi','.mkv'}
SCHEMA="""
CREATE TABLE IF NOT EXISTS vision_videos (
 video_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, processed_utc TEXT NOT NULL,
 fps REAL, width INTEGER, height INTEGER, total_frames INTEGER,
 frames_sampled INTEGER, detections INTEGER, model TEXT, status TEXT,
 output_video TEXT, error TEXT, data_quality TEXT
);
CREATE TABLE IF NOT EXISTS vision_observations (
 video_id TEXT NOT NULL, frame INTEGER NOT NULL, seconds REAL NOT NULL,
 track_id INTEGER NOT NULL, category TEXT NOT NULL, confidence REAL NOT NULL,
 x1 REAL,y1 REAL,x2 REAL,y2 REAL,center_x REAL,center_y REAL,
 PRIMARY KEY(video_id,frame,track_id,category)
);
CREATE INDEX IF NOT EXISTS vision_obs_track ON vision_observations(video_id,track_id,seconds);
CREATE TABLE IF NOT EXISTS vision_metrics (
 video_id TEXT NOT NULL, track_id INTEGER NOT NULL, category TEXT NOT NULL,
 observations INTEGER NOT NULL, visible_seconds REAL NOT NULL,
 distance_pixels REAL NOT NULL, mean_speed_px_s REAL, max_speed_px_s REAL,
 PRIMARY KEY(video_id,track_id,category)
);
CREATE TABLE IF NOT EXISTS vision_events (
 video_id TEXT NOT NULL, frame INTEGER NOT NULL, seconds REAL NOT NULL,
 event_type TEXT NOT NULL, actor_track INTEGER, other_track INTEGER,
 confidence REAL, details_json TEXT,
 PRIMARY KEY(video_id,frame,event_type,actor_track,other_track)
);
"""
def connect(db_path=DB):
    p=Path(db_path);p.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(p,timeout=30)
    c.executescript(SCHEMA)
    return c

def video_id_for(path):
    """Content hash avoids duplicate imports under renamed filenames."""
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()[:24]

def movement_metrics(observations, max_gap_s=0.6):
    """Pixel movement is NOT court-calibrated distance or physical speed."""
    by_track=defaultdict(list)
    for o in observations:
        if o['category']=='person':by_track[o['track_id']].append(o)
    result=[]
    for tid, rows in by_track.items():
        rows.sort(key=lambda x:x['seconds'])
        distance=0.0; velocities=[]; valid_time=0.0
        for a,b in zip(rows,rows[1:]):
            dt=b['seconds']-a['seconds']
            if dt<=0 or dt>max_gap_s:continue
            d=((b['center_x']-a['center_x'])**2+(b['center_y']-a['center_y'])**2)**0.5
            distance+=d;valid_time+=dt;velocities.append(d/dt)
        result.append(dict(track_id=tid,category='person',observations=len(rows),
                           visible_seconds=valid_time,distance_pixels=distance,
                           mean_speed_px_s=distance/valid_time if valid_time else None,
                           max_speed_px_s=max(velocities) if velocities else None))
    return result

def ball_proximity_events(observations, threshold_fraction=0.07, frame_width=1280):
    """Ball-nearest-person is a heuristic, NOT verified possession or pass."""
    by_frame=defaultdict(list)
    for o in observations:by_frame[o['frame']].append(o)
    events=[];prev=None
    for frame,rows in sorted(by_frame.items()):
        balls=[x for x in rows if x['category']=='sports ball']
        people=[x for x in rows if x['category']=='person']
        if not balls or not people:continue
        ball=max(balls,key=lambda x:x['confidence'])
        distances=sorted((((p['center_x']-ball['center_x'])**2+(p['center_y']-ball['center_y'])**2)**.5,p) for p in people)
        dist,p=distances[0]
        if dist>threshold_fraction*frame_width:prev=None;continue
        if prev!=p['track_id']:
            events.append(dict(frame=frame,seconds=ball['seconds'],event_type='ball_nearest_person_changed',
                               actor_track=p['track_id'],other_track=prev,
                               confidence=min(ball['confidence'],p['confidence']),
                               details_json=json.dumps({'distance_px':round(dist,2),'note':'proximity heuristic; not a verified pass or possession'})))
        prev=p['track_id']
    return events

def process_video(path, db_path=DB, out_dir=OUT, model_name='yolo11n.pt', stride=2,
                  confidence=.28, max_frames=0, progress=None, annotate=True):
    """Run local detector/tracker and persist *observations*, not guessed stats."""
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Video tracking dependencies missing. Install with: pip install ultralytics opencv-python-headless') from exc
    path=Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in SUPPORTED:raise ValueError('Select a local video file (MP4, MOV, M4V, AVI, MKV).')
    if stride<1:raise ValueError('stride must be at least 1')
    vid=video_id_for(path)
    with connect(db_path) as c:
        exists=c.execute('SELECT status FROM vision_videos WHERE video_id=?',(vid,)).fetchone()
        if exists and exists[0]=='complete':return {'video_id':vid,'status':'already_processed','frames_sampled':0}
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError('Could not open video')
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    video_path=out_dir/(vid+'_tracked.mp4')
    writer=None
    if annotate:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        writer=cv2.VideoWriter(str(video_path),fourcc,max(1.0,fps/stride),(width,height))
        if not writer.isOpened():
            cap.release();raise RuntimeError('Could not create annotated output video')
    model=YOLO(model_name)  # model download may require internet first time
    observations=[];frame=0;sampled=0
    try:
        while True:
            ok,img=cap.read()
            if not ok:break
            frame+=1
            if (frame-1)%stride:continue
            if max_frames and sampled>=max_frames:break
            sampled+=1
            result=model.track(img,persist=True,tracker='bytetrack.yaml',conf=confidence,verbose=False,classes=[0,32])[0]
            if result.boxes is not None and len(result.boxes):
                boxes=result.boxes
                for i in range(len(boxes)):
                    cls=int(boxes.cls[i].item())
                    category='person' if cls==0 else 'sports ball' if cls==32 else 'other'
                    if category=='other':continue
                    tid=int(boxes.id[i].item()) if boxes.id is not None else -(i+1)
                    x1,y1,x2,y2=[float(x) for x in boxes.xyxy[i].tolist()]
                    conf=float(boxes.conf[i].item())
                    observations.append(dict(frame=frame,seconds=(frame-1)/fps,track_id=tid,
                        category=category,confidence=conf,x1=x1,y1=y1,x2=x2,y2=y2,
                        center_x=(x1+x2)/2,center_y=y2 if category=='person' else (y1+y2)/2))
                    if writer:
                        color=(20,225,180) if category=='person' else (20,150,255)
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                        cv2.putText(img,f'{category} #{tid} {conf:.2f}',(int(x1),max(15,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            if writer:writer.write(img)
            if progress and sampled%25==0:progress(sampled, max(1,(total+stride-1)//stride))
    finally:
        cap.release()
        if writer:writer.release()
    metrics=movement_metrics(observations,max_gap_s=max(.6,2.2*stride/fps))
    events=ball_proximity_events(observations,frame_width=width)
    with connect(db_path) as c:
        c.execute('DELETE FROM vision_observations WHERE video_id=?',(vid,))
        c.execute('DELETE FROM vision_metrics WHERE video_id=?',(vid,))
        c.execute('DELETE FROM vision_events WHERE video_id=?',(vid,))
        c.execute("""INSERT OR REPLACE INTO vision_videos
            (video_id,source_path,processed_utc,fps,width,height,total_frames,frames_sampled,
             detections,model,status,output_video,data_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (vid,str(path),datetime.now(timezone.utc).isoformat(),fps,width,height,total,sampled,
             len(observations),model_name,'complete',str(video_path) if writer else None,
             'Uncalibrated broadcast video; generic person/ball detection; no team/identity or physical distances.'))
        c.executemany("""INSERT INTO vision_observations
            (video_id,frame,seconds,track_id,category,confidence,x1,y1,x2,y2,center_x,center_y)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(vid,*[o[k] for k in ('frame','seconds','track_id','category','confidence','x1','y1','x2','y2','center_x','center_y')]) for o in observations])
        c.executemany("""INSERT INTO vision_metrics
            (video_id,track_id,category,observations,visible_seconds,distance_pixels,mean_speed_px_s,max_speed_px_s)
            VALUES (?,?,?,?,?,?,?,?)""",
            [(vid,*[m[k] for k in ('track_id','category','observations','visible_seconds','distance_pixels','mean_speed_px_s','max_speed_px_s')]) for m in metrics])
        c.executemany("""INSERT OR REPLACE INTO vision_events
            (video_id,frame,seconds,event_type,actor_track,other_track,confidence,details_json)
            VALUES (?,?,?,?,?,?,?,?)""",
            [(vid,*[e[k] for k in ('frame','seconds','event_type','actor_track','other_track','confidence','details_json')]) for e in events])
    return {'video_id':vid,'status':'complete','frames_sampled':sampled,'observations':len(observations),
            'track_count':len(metrics),'proximity_changes':len(events),'output_video':str(video_path) if writer else None}

def watch_folder(folder=INBOX, interval=120, **kwargs):
    """Unattended processing of local videos while this process/Mac is running."""
    folder=Path(folder).expanduser();folder.mkdir(parents=True,exist_ok=True)
    print('COURTSIDE VISION watching:',folder,flush=True)
    while True:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED or not path.is_file():continue
            try:
                if time.time()-path.stat().st_mtime<30:continue # avoid partially copied files
                result=process_video(path,**kwargs)
                if result['status']!='already_processed':print(result,flush=True)
            except Exception as exc:print(f'{path.name}: {type(exc).__name__}: {exc}',flush=True)
        time.sleep(interval)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Analyze local basketball video with COURTSIDE VISION')
    ap.add_argument('video',nargs='?',help='Local video path')
    ap.add_argument('--watch',action='store_true',help='Process new videos in inbox continuously while Mac is awake')
    ap.add_argument('--stride',type=int,default=2)
    ap.add_argument('--model',default='yolo11n.pt')
    ap.add_argument('--max-frames',type=int,default=0)
    args=ap.parse_args()
    if args.watch:watch_folder(stride=args.stride,model_name=args.model,max_frames=args.max_frames)
    elif args.video:print(json.dumps(process_video(args.video,stride=args.stride,model_name=args.model,max_frames=args.max_frames),indent=2))
    else:ap.error('Provide a video or --watch')
