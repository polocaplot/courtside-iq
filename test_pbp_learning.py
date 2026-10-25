import json
import sqlite3
from datetime import date
from pathlib import Path
import tempfile
import pandas as pd
from nba_archive import connect,apply_archive_priors
from pbp_learning import (setup,extract_video_refs,index_event_video_metadata,
                          index_local_event_clips,game_coverage,player_shot_profile,team_event_profile)

def make_db(path):
    db=connect(path)
    setup(db)
    for i in range(1,7):
        gid=f'G{i}';day=f'2026-10-{i:02d}'
        db.execute('''INSERT INTO games(game_id,game_date,home,away,state,updated_utc)
                     VALUES(?,?,?,?,?,?)''',(gid,day,'ATL','HOU','post','now'))
        db.execute('''INSERT INTO player_games(game_id,player_id,team,player_name,fga,fg3a,minutes,pf,played,starter)
                     VALUES(?,?,?,?,?,?,?,?,?,?)''',(gid,'7','ATL','Example Player',10,4,30,2,1,1))
        db.execute('INSERT INTO team_games(game_id,team,fga,pace_est) VALUES(?,?,?,?)',(gid,'ATL',10,99))
        for j in range(10):
            pid=f'p{j}';is3=int(j<4)
            db.execute('''INSERT INTO plays(game_id,play_id,sequence,team,athlete_id,raw_json)
                          VALUES(?,?,?,?,?,?)''',(gid,pid,j,'ATL','7',json.dumps({'video':{'url':'https://example.com/clip.mp4'}} if j==0 else {})))
            db.execute('INSERT INTO shots(game_id,play_id,team,athlete_id) VALUES(?,?,?,?)',(gid,pid,'ATL','7'))
            db.execute('INSERT INTO play_features(game_id,play_id,category,is_three) VALUES(?,?,?,?)',(gid,pid,'shot',is3))
    db.commit()
    return db

def test_complete_game_events_and_video_index(tmp_path):
    with make_db(tmp_path/'nba.db') as db:
        assert extract_video_refs({'video':{'url':'https://example.com/clip.mp4'}})
        assert index_event_video_metadata(db,'G1')==1
        cov=game_coverage(db,'G1')
        assert cov['events']==10 and cov['shot_event_coverage']==1.0
        assert cov['whole_game_video'] is False and cov['local_video_events']==0
        clips=tmp_path/'clips';clips.mkdir();(clips/'G1__p0.mp4').write_bytes(b'fixture')
        assert index_local_event_clips(db,clips)=={'linked':1,'unmatched':0}
        assert game_coverage(db,'G1')['local_video_events']==1
        assert index_local_event_clips(db,clips)['linked']==1

def test_asof_shot_learning_and_no_future_leak(tmp_path):
    with make_db(tmp_path/'nba.db') as db:
        assert player_shot_profile(db,'ATL','Example Player','2026-10-04') is None
        prof=player_shot_profile(db,'ATL','Example Player','2026-10-07')
        assert prof and prof['games']==6 and abs(prof['three_attempt_rate']-.4)<1e-8
        assert team_event_profile(db,'ATL','2026-10-04')['games']==3
        assert team_event_profile(db,'ATL','2026-10-07')['games']==6
        db.execute("DELETE FROM shots WHERE game_id='G1' AND play_id<>'p0'")
        assert player_shot_profile(db,'ATL','Example Player','2026-10-05') is None

def test_simulator_pbp_prior_only_uses_prior_games(tmp_path):
    dbpath=tmp_path/'nba.db'
    with make_db(dbpath):pass
    teams=pd.DataFrame([{'TEAM_ABBREVIATION':'ATL','PACE':101.}])
    players=pd.DataFrame([{'TEAM_ABBREVIATION':'ATL','PLAYER_NAME':'Example Player','MIN':32.,
                           'PF':2.,'FGA':10.,'FG3A':8.,'FG3M':3.2}])
    t,p,info=apply_archive_priors(teams,players,'2026-10-07',db_path=dbpath)
    assert info['pbp_shot_profiles']==1
    assert 0<p.iloc[0]['FG3A']<8
    assert p.iloc[0]['PBP_SHOT_PROFILE_GAMES']==6
    t,p,info=apply_archive_priors(teams,players,'2026-10-04',db_path=dbpath)
    assert info['pbp_shot_profiles']==0
    assert p.iloc[0]['FG3A']==8
