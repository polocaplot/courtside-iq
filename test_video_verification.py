import io,json,sqlite3
from automatic_video import verify_candidate,search_youtube,reverify_existing,setup

def v(title,published='2026-10-24T10:00:00Z'):
    return {'title':title,'published':published}

def test_correct_match_and_reject_wrong_date():
    assert verify_candidate(v('Hawks vs Rockets | Full Game Highlights'),'HOU','ATL','2026-10-24')[0]=='verified_metadata'
    assert verify_candidate(v('Hawks vs Rockets | Full Game Highlights', '2026-03-03T10:00:00Z'),'HOU','ATL','2026-10-24')[0]=='rejected_mismatch'
    assert verify_candidate(v('Hawks vs Rockets | March 3, 2026', '2026-10-24T10:00:00Z'),'HOU','ATL','2026-10-24')[0]=='rejected_mismatch'
    assert verify_candidate(v('Lakers vs Clippers | Full Game Highlights'),'HOU','ATL','2026-10-24')[0]=='rejected_mismatch'
    assert verify_candidate(v('Hawks vs Rockets | Summer League'),'HOU','ATL','2026-10-24')[0]=='rejected_mismatch'
    assert verify_candidate(v('Hawks vs Rockets', ''),'HOU','ATL','2026-10-24')[0]=='candidate_unverified'

def test_durations_and_metadata_only():
    class Response(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*a):self.close()
    def opener(req,timeout):
        if '/videos?' in req.full_url:
            return Response(json.dumps({'items':[{'id':'full','contentDetails':{'duration':'PT1H55M'}},
                {'id':'short','contentDetails':{'duration':'PT9M30S'}}]}).encode())
        return Response(json.dumps({'items':[{'id':{'videoId':i},'snippet':{
            'title':'Houston Rockets vs Atlanta Hawks | Full Game Highlights',
            'publishedAt':'2026-10-25T04:00:00Z'}} for i in ['full','short']]}).encode())
    found=search_youtube('g','HOU','ATL','2026-10-24','key',opener)
    assert [x['video_kind'] for x in found]==['possible_full_game','highlight_or_clip']
    assert all(x['status']=='verified_metadata' for x in found)

def test_reverify_previous_bad_rows():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE games(game_id TEXT,game_date TEXT,away TEXT,home TEXT,state TEXT)')
    db.execute("INSERT INTO games VALUES('g1','2026-10-24','HOU','ATL','post')")
    setup(db)
    db.execute("INSERT INTO video_discoveries(game_id,provider,video_id,title,published) VALUES('g1','youtube','old','Lakers vs Clippers March 2026','2026-03-03T00:00:00Z')")
    assert reverify_existing(db)['rejected_mismatch']==1
    assert db.execute('SELECT status FROM video_discoveries').fetchone()[0]=='rejected_mismatch'
