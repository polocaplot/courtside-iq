import io,json,sqlite3
from automatic_video import setup,search_youtube,discover

def test_search_metadata_only():
    class Response(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
    def opener(req,timeout):
        assert 'youtube/v3/search' in req.full_url
        return Response(json.dumps({'items':[{'id':{'videoId':'abc123'},'snippet':{'title':'Highlights','channelTitle':'NBA'}}]}).encode())
    videos=search_youtube('g1','ATL','HOU','2026-10-24','test',opener)
    assert len(videos)==1 and videos[0]['video_id']=='abc123'
    assert 'local_path' not in videos[0]

def test_discovery_idempotent():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE games(game_id TEXT,game_date TEXT,away TEXT,home TEXT,state TEXT)')
    db.execute("INSERT INTO games VALUES('g1','2026-10-24','ATL','HOU','post')")
    class Response(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
    def opener(req,timeout):return Response(b'{"items":[{"id":{"videoId":"abc"},"snippet":{"title":"Clip"}}]}')
    assert discover(db,'test',opener=opener)['found']==1
    assert discover(db,'test',opener=opener)['found']==1
    assert db.execute('SELECT count(*) FROM video_discoveries').fetchone()[0]==1
    assert discover(db,'')['status']=='disabled'
