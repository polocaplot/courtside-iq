import sys,tempfile,sqlite3
from pathlib import Path
sys.path.insert(0,'/mnt/data/ciq_vision_build/COURTSIDE IQ.app/Contents/Resources/source')
import courtside_vision as v

def test_db():
 with tempfile.TemporaryDirectory() as d:
  with v.connect(Path(d)/'test.db') as c:
   for t in ('vision_videos','vision_observations','vision_metrics','vision_events'):
    assert c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",(t,)).fetchone()

def test_id():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'a.mp4';q=Path(d)/'b.mp4';p.write_bytes(b'video');q.write_bytes(b'video')
  assert v.video_id_for(p)==v.video_id_for(q)
  q.write_bytes(b'other');assert v.video_id_for(p)!=v.video_id_for(q)

def test_metrics():
 rows=[{'category':'person','track_id':2,'seconds':0,'center_x':0,'center_y':0},
 {'category':'person','track_id':2,'seconds':.2,'center_x':3,'center_y':4},
 {'category':'person','track_id':2,'seconds':2,'center_x':100,'center_y':100}]
 x=v.movement_metrics(rows)[0];assert x['distance_pixels']==5 and x['mean_speed_px_s']==25

def test_proximity():
 rows=[]
 for frame,ballx in [(1,10),(2,20)]:
  rows.extend([dict(frame=frame,seconds=frame/30,track_id=100,category='sports ball',confidence=.9,center_x=ballx,center_y=10),
  dict(frame=frame,seconds=frame/30,track_id=1,category='person',confidence=.8,center_x=10,center_y=10),
  dict(frame=frame,seconds=frame/30,track_id=2,category='person',confidence=.8,center_x=20,center_y=10)])
 events=v.ball_proximity_events(rows,frame_width=200)
 assert len(events)==2 and events[0]['actor_track']==1 and events[1]['actor_track']==2

if __name__=='__main__':
 for f in (test_db,test_id,test_metrics,test_proximity):f();print('PASS',f.__name__)
