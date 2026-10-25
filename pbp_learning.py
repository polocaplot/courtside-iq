"""Evidence-driven full-game play-by-play learning and optional local event video links.

No assumption that PBP videos cover a complete game. No remote video scraping.
As-of joins prohibit future-game leakage. Video tracking remains experimental.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sqlite3
import json

SCHEMA="""
CREATE TABLE IF NOT EXISTS event_videos (
 game_id TEXT NOT NULL, play_id TEXT NOT NULL, video_ref TEXT NOT NULL,
 source TEXT NOT NULL, local_path TEXT, vision_video_id TEXT,
 status TEXT NOT NULL DEFAULT 'metadata_only', checked_utc TEXT,
 PRIMARY KEY(game_id,play_id,video_ref)
);
CREATE INDEX IF NOT EXISTS ix_event_videos_game ON event_videos(game_id,play_id);
"""
VIDEO_EXTS={'.mp4','.mov','.m4v','.avi','.mkv'}

def setup(db):
    for statement in SCHEMA.split(";"):
        if statement.strip(): db.execute(statement)

def extract_video_refs(raw):
    """Preserve *only* explicit event video metadata; don't invent clip URLs."""
    refs=set()
    for key in ('video','videos','highlight','highlights'):
        value=raw.get(key)
        if not value:continue
        items=value if isinstance(value,list) else [value]
        for item in items:
            if isinstance(item,str) and item.startswith('https://'):
                refs.add(item)
            if isinstance(item,dict):
                for field in ('href','url','source','link'):
                    candidate=item.get(field)
                    if isinstance(candidate,str) and candidate.startswith('https://'):
                        refs.add(candidate)
    return sorted(refs)

def index_event_video_metadata(db,game_id):
    """Metadata only: URLs are NOT downloaded or assumed playable/licensed."""
    setup(db)
    rows=db.execute('SELECT play_id,raw_json FROM plays WHERE game_id=?',(str(game_id),)).fetchall()
    count=0
    for play_id,raw in rows:
        try: refs=extract_video_refs(json.loads(raw))
        except (ValueError,TypeError):continue
        for url in refs:
            db.execute('''INSERT OR IGNORE INTO event_videos
                (game_id,play_id,video_ref,source,status,checked_utc)
                VALUES(?,?,?,?,?,?)''',(str(game_id),str(play_id),url,'play-by-play provider','metadata_only',
                      datetime.now(timezone.utc).isoformat()))
            count+=1
    return count

def index_local_event_clips(db,folder):
    """Accept filename GAMEID__PLAYID.mp4 for footage the user may analyze.

    The link is rejected if the event doesn't exist in the archive.
    """
    setup(db)
    folder=Path(folder).expanduser()
    if not folder.exists():return {'linked':0,'unmatched':0}
    linked=unmatched=0
    for file in sorted(folder.iterdir()):
        if not file.is_file() or file.suffix.lower() not in VIDEO_EXTS:continue
        parts=file.stem.split('__',1)
        if len(parts)!=2:unmatched+=1;continue
        game_id,play_id=parts
        exists=db.execute('SELECT 1 FROM plays WHERE game_id=? AND play_id=?',(game_id,play_id)).fetchone()
        if not exists:unmatched+=1;continue
        db.execute('''INSERT INTO event_videos(game_id,play_id,video_ref,source,local_path,status,checked_utc)
                      VALUES(?,?,?,?,?,?,?) ON CONFLICT(game_id,play_id,video_ref) DO UPDATE SET
                      local_path=excluded.local_path,status=excluded.status,checked_utc=excluded.checked_utc''',
                   (game_id,play_id,file.resolve().as_uri(),'user-authorized local footage',str(file.resolve()),
                    'local_available',datetime.now(timezone.utc).isoformat()))
        linked+=1
    db.commit()
    return {'linked':linked,'unmatched':unmatched}

def game_coverage(db,game_id):
    setup(db)
    gid=str(game_id)
    q=lambda sql: db.execute(sql,(gid,)).fetchone()[0]
    events=q('SELECT COUNT(*) FROM plays WHERE game_id=?')
    shots=q('SELECT COUNT(*) FROM shots WHERE game_id=?')
    box=q('SELECT COALESCE(SUM(fga),0) FROM team_games WHERE game_id=?')
    vids=q('SELECT COUNT(DISTINCT play_id) FROM event_videos WHERE game_id=?')
    local=q("SELECT COUNT(DISTINCT play_id) FROM event_videos WHERE game_id=? AND local_path IS NOT NULL")
    return {'game_id':gid,'events':events,'shot_events':shots,'box_fga':box,
            'shot_event_coverage':min(1.,shots/box) if box else None,
            'video_linked_events':vids,'local_video_events':local,
            'local_video_event_fraction':local/events if events else 0.,
            'whole_game_video':False,
            'note':'Play-by-play is an event timeline, not continuous video. Event clips cannot establish whole-game spatial coverage.'}

def player_shot_profile(db,team,player_name,cutoff,lookback=25):
    """Learn a 3-point shot-selection tendency from complete-enough *past* PBP.

    Match by provider athlete ID, verify that play-by-play shot events align with
    box-score FGA, and discard unknown 2pt/3pt classifications.
    """
    rows=db.execute('''SELECT pg.game_id,pg.fga AS box_fga,pg.fg3a AS box_fg3a,
                COUNT(s.play_id) AS event_fga,
                SUM(CASE WHEN pf.is_three IS NOT NULL THEN 1 ELSE 0 END) AS classified,
                SUM(CASE WHEN pf.is_three=1 THEN 1 ELSE 0 END) AS event_fg3a
            FROM player_games pg JOIN games g ON g.game_id=pg.game_id
            LEFT JOIN shots s ON s.game_id=pg.game_id AND s.athlete_id=pg.player_id
            LEFT JOIN play_features pf ON pf.game_id=s.game_id AND pf.play_id=s.play_id
            WHERE pg.team=? AND lower(pg.player_name)=lower(?) AND g.game_date<?
              AND g.state='post' AND pg.fga>0
            GROUP BY pg.game_id,pg.player_id ORDER BY g.game_date DESC LIMIT ?''',
            (team,player_name,cutoff,lookback)).fetchall()
    eligible=[]
    for r in rows:
        fga=float(r['box_fga'] or 0)
        event=float(r['event_fga'] or 0)
        classified=float(r['classified'] or 0)
        # Refuse to learn from incomplete PBP or unknown 3-point classifications.
        if fga>0 and event/fga>=.80 and event/fga<=1.15 and classified/max(event,1)>=.85:
            eligible.append(r)
    total=sum(float(r['classified'] or 0) for r in eligible)
    if len(eligible)<4 or total<40:return None
    threes=sum(float(r['event_fg3a'] or 0) for r in eligible)
    return {'games':len(eligible),'classified_shots':int(total),
            'three_attempt_rate':threes/total,'cutoff_exclusive':cutoff,
            'source':'past play-by-play shots validated against player boxscore',
            'unverified_games':len(rows)-len(eligible)}

def team_event_profile(db,team,cutoff,lookback=20):
    """Past-only game-level foul/turnover/shot events; observational, not tracking."""
    rows=db.execute('''SELECT g.game_id,g.game_date,
                SUM(CASE WHEN pf.category='shot' THEN 1 ELSE 0 END) AS shots,
                SUM(CASE WHEN pf.category='turnover' THEN 1 ELSE 0 END) AS turnovers,
                SUM(CASE WHEN pf.category='foul' THEN 1 ELSE 0 END) AS fouls,
                SUM(CASE WHEN pf.category='rebound' THEN 1 ELSE 0 END) AS rebounds,
                COUNT(*) AS events
            FROM plays p JOIN games g USING(game_id)
            JOIN play_features pf ON pf.game_id=p.game_id AND pf.play_id=p.play_id
            WHERE p.team=? AND g.game_date<? AND g.state='post'
            GROUP BY g.game_id ORDER BY g.game_date DESC LIMIT ?''',(team,cutoff,lookback)).fetchall()
    if not rows:return None
    totals={k:sum(int(r[k] or 0) for r in rows) for k in ('shots','turnovers','fouls','rebounds','events')}
    return {'games':len(rows),'cutoff_exclusive':cutoff,**totals,
            'events_per_game':totals['events']/len(rows),
            'note':'Observed event counts, not verified possession boundaries or optical tracking.'}
