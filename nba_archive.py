"""COURTSIDE IQ NBA archive: SQLite, source-preserving, resumable, no fabricated tracking.

Public ESPN scoreboard and summary are unofficial. Full optical tracking, exact
individual defender assignments, and some event labels are not publicly provided.
Store raw responses so new derived metrics can be rebuilt without re-downloading.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib
from typing import Any

from data_sources import get_json, schedule, ESPN, injuries

DB_PATH = Path.home() / '.courtsideiq' / 'nba_archive.sqlite3'
SOURCE = 'ESPN unofficial scoreboard/summary'

SCHEMA = '''
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS games (
 game_id TEXT PRIMARY KEY, game_date TEXT NOT NULL, season TEXT,
 tipoff_utc TEXT, home TEXT, away TEXT, home_score INTEGER, away_score INTEGER,
 state TEXT, status TEXT, periods INTEGER, neutral_site INTEGER,
 venue TEXT, attendance INTEGER, source TEXT, updated_utc TEXT NOT NULL,
 detail_status TEXT DEFAULT 'pending', detail_error TEXT
);
CREATE INDEX IF NOT EXISTS ix_games_date ON games(game_date,state);
CREATE TABLE IF NOT EXISTS team_games (
 game_id TEXT, team TEXT, opponent TEXT, home INTEGER,
 minutes REAL, points REAL, fgm REAL, fga REAL, fg3m REAL, fg3a REAL,
 ftm REAL, fta REAL, oreb REAL, dreb REAL, reb REAL, ast REAL,
 stl REAL, blk REAL, tov REAL, pf REAL,
 possessions_est REAL, pace_est REAL, off_rating_est REAL,
 def_rating_est REAL, net_rating_est REAL, efg_pct REAL, ts_pct REAL,
 tov_pct REAL, oreb_pct REAL, ft_rate REAL, three_rate REAL,
 boxscore_complete INTEGER DEFAULT 0, raw_json TEXT,
 PRIMARY KEY(game_id,team), FOREIGN KEY(game_id) REFERENCES games(game_id)
);
CREATE INDEX IF NOT EXISTS ix_team_games_team ON team_games(team,game_id);
CREATE TABLE IF NOT EXISTS player_games (
 game_id TEXT, player_id TEXT, team TEXT, player_name TEXT, starter INTEGER,
 played INTEGER, minutes REAL, points REAL, fgm REAL, fga REAL, fg3m REAL,
 fg3a REAL, ftm REAL, fta REAL, oreb REAL, dreb REAL, reb REAL, ast REAL,
 stl REAL, blk REAL, tov REAL, pf REAL, plus_minus REAL, raw_json TEXT,
 PRIMARY KEY(game_id,player_id,team), FOREIGN KEY(game_id) REFERENCES games(game_id)
);
CREATE INDEX IF NOT EXISTS ix_player_history ON player_games(player_id,game_id);
CREATE INDEX IF NOT EXISTS ix_player_team ON player_games(team,player_name);
CREATE TABLE IF NOT EXISTS plays (
 game_id TEXT, play_id TEXT, sequence INTEGER, period INTEGER,
 clock TEXT, clock_seconds REAL, team TEXT, athlete_id TEXT,
 event_type TEXT, description TEXT, scoring INTEGER, score_value INTEGER,
 home_score INTEGER, away_score INTEGER, shooting INTEGER,
 x REAL, y REAL, raw_json TEXT NOT NULL,
 PRIMARY KEY(game_id,play_id), FOREIGN KEY(game_id) REFERENCES games(game_id)
);
CREATE INDEX IF NOT EXISTS ix_plays_game_order ON plays(game_id,sequence);
CREATE TABLE IF NOT EXISTS shots (
 game_id TEXT, play_id TEXT, period INTEGER, clock_seconds REAL,
 team TEXT, athlete_id TEXT, made INTEGER, points INTEGER,
 x REAL, y REAL, distance_ft REAL, shot_kind TEXT,
 PRIMARY KEY(game_id,play_id), FOREIGN KEY(game_id,play_id) REFERENCES plays(game_id,play_id)
);
CREATE TABLE IF NOT EXISTS substitutions (
 game_id TEXT, play_id TEXT, period INTEGER, clock_seconds REAL,
 team TEXT, athlete_id TEXT, description TEXT, raw_json TEXT,
 PRIMARY KEY(game_id,play_id)
);
CREATE TABLE IF NOT EXISTS period_scores (
 game_id TEXT, team TEXT, period INTEGER, points INTEGER,
 PRIMARY KEY(game_id,team,period)
);
CREATE TABLE IF NOT EXISTS source_payloads (
 game_id TEXT, kind TEXT, fetched_utc TEXT, compressed_json BLOB,
 PRIMARY KEY(game_id,kind)
);
CREATE TABLE IF NOT EXISTS injury_snapshots (
 snapshot_date TEXT, team_id TEXT, player_name TEXT, status TEXT,
 note TEXT, checked_utc TEXT,
 PRIMARY KEY(snapshot_date,team_id,player_name)
);
CREATE TABLE IF NOT EXISTS coverage (
 game_id TEXT, category TEXT, available INTEGER, note TEXT,
 PRIMARY KEY(game_id,category)
);
CREATE TABLE IF NOT EXISTS game_team_context (
 game_id TEXT, team TEXT, rest_days INTEGER, back_to_back INTEGER,
 opponent_efg_pct REAL, opponent_three_rate REAL,
 opponent_ft_rate REAL, opponent_tov_pct REAL,
 opponent_oreb_pct REAL, opponent_points REAL,
 PRIMARY KEY(game_id,team)
);
CREATE TABLE IF NOT EXISTS play_features (
 game_id TEXT, play_id TEXT, category TEXT, subtype TEXT,
 elapsed_seconds REAL, score_margin_home INTEGER, clutch INTEGER,
 shot_zone TEXT, is_three INTEGER, source_certainty TEXT,
 PRIMARY KEY(game_id,play_id)
);
CREATE TABLE IF NOT EXISTS sync_state (key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS sync_errors (
 day TEXT, game_id TEXT, error TEXT, checked_utc TEXT,
 PRIMARY KEY(day,game_id)
);
'''

def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def connect(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    db=sqlite3.connect(str(path),timeout=35)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA busy_timeout=35000')
    db.executescript(SCHEMA)
    return db

def number(v, default=None):
    if v is None or v == '' or v == '--': return default
    try: return float(str(v).replace(',','').replace('+',''))
    except (ValueError,TypeError): return default

def integer(v, default=None):
    n=number(v)
    return int(n) if n is not None else default

def minutes(v):
    if v is None: return None
    if isinstance(v,(int,float)): return float(v)
    s=str(v).strip()
    if ':' in s:
        try:
            a,b=s.split(':',1)
            return int(a)+float(b)/60
        except ValueError: return None
    return number(s)

def clock_secs(clock):
    if isinstance(clock,dict):
        if clock.get('value') is not None: return number(clock['value'])
        clock=clock.get('displayValue','')
    if isinstance(clock,str) and ':' in clock:
        try:
            a,b=clock.split(':',1)
            return 60*int(a)+float(b)
        except ValueError: pass
    return number(clock)

def season_for(day: date):
    start=day.year if day.month>=7 else day.year-1
    return f'{start}-{str(start+1)[-2:]}'

def _store_raw(db, game_id, kind, data):
    packed=zlib.compress(json.dumps(data,separators=(',',':'),ensure_ascii=False).encode(),6)
    db.execute('INSERT OR REPLACE INTO source_payloads VALUES(?,?,?,?)',(game_id,kind,utc(),packed))

def _upsert_game(db, g, day):
    home,away=g['home'],g['away']
    db.execute('''INSERT INTO games(game_id,game_date,season,tipoff_utc,home,away,
       home_score,away_score,state,status,source,updated_utc)
       VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
       home_score=excluded.home_score,away_score=excluded.away_score,
       state=excluded.state,status=excluded.status,updated_utc=excluded.updated_utc''',
       (str(g['id']),str(day),season_for(day),g.get('tipoff'),home.get('abbr'),away.get('abbr'),
        integer(home.get('score')),integer(away.get('score')),g.get('state'),g.get('status'),SOURCE,utc()))

def _team_abbr(t):
    team=t.get('team') or {}
    return team.get('abbreviation') or team.get('shortDisplayName') or ''

def _stat_dict(stats):
    if isinstance(stats,dict):
        return {str(k).lower():v for k,v in stats.items()}
    result={}
    for s in stats or []:
        if not isinstance(s,dict): continue
        key=str(s.get('name') or s.get('abbreviation') or s.get('label') or '').lower()
        if key: result[key]=s.get('displayValue',s.get('value'))
    return result

def _split_made_attempts(v):
    if isinstance(v,str) and '-' in v:
        a,b=v.split('-',1)
        return number(a),number(b)
    return None,None

def _box_numbers(stats):
    """Accept both ESPN summary player label arrays and team stat dictionaries."""
    s=_stat_dict(stats)
    aliases={
      'points':['points','pts'], 'reb':['totalrebounds','rebounds','reb'],
      'ast':['assists','ast'], 'stl':['steals','stl'], 'blk':['blocks','blk'],
      'tov':['turnovers','to','tov'], 'pf':['fouls','personalfouls','pf'],
      'oreb':['offensiverebounds','offreb','oreb'],
      'dreb':['defensiverebounds','defreb','dreb'],
      'fgm':['fieldgoalsmade','fgm'],'fga':['fieldgoalsattempted','fga'],
      'fg3m':['threepointfieldgoalsmade','threepointmade','3pm','fg3m'],
      'fg3a':['threepointfieldgoalsattempted','threepointattempted','3pa','fg3a'],
      'ftm':['freethrowsmade','ftm'],'fta':['freethrowsattempted','fta'],
    }
    out={k:next((number(s[key]) for key in keys if key in s and number(s[key]) is not None),None)
         for k,keys in aliases.items()}
    for label,keys in [('fg',['fieldgoals','fieldgoalsmade-fieldgoalsattempted','fg']),('fg3',['threepointfieldgoals','threepointfieldgoalsmade-threepointfieldgoalsattempted','threepointers','3pt']),('ft',['freethrows','freethrowsmade-freethrowsattempted','ft'])]:
        val=next((s[k] for k in keys if k in s),None)
        a,b=_split_made_attempts(val)
        if a is not None:
            out[label+'m']=a;out[label+'a']=b
    out['minutes']=minutes(s.get('minutes',s.get('min')))
    out['plus_minus']=number(s.get('plusminus',s.get('+/-')))
    return out

def _parse_summary_box(db, gid, payload, home, away):
    box=payload.get('boxscore') or {}
    team_list=box.get('teams') or []
    player_list=box.get('players') or []
    counts={'teams':0,'players':0}
    team_stats={}
    for entry in team_list:
        abbr=_team_abbr(entry)
        if not abbr: continue
        n=_box_numbers(entry.get('statistics') or [])
        team_stats[abbr]=n
        cols=['points','fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','reb','ast','stl','blk','tov','pf']
        db.execute('''INSERT OR REPLACE INTO team_games(game_id,team,opponent,home,minutes,
          points,fgm,fga,fg3m,fg3a,ftm,fta,oreb,dreb,reb,ast,stl,blk,tov,pf,boxscore_complete,raw_json)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
          (gid,abbr,away if abbr==home else home,int(abbr==home),
           48.,*[n[c] for c in cols],int(n['fga'] is not None and n['tov'] is not None),json.dumps(entry)))
        counts['teams']+=1
    for team_entry in player_list:
        abbr=_team_abbr(team_entry)
        if not abbr: continue
        for group in team_entry.get('statistics') or []:
            labels=[str(x).lower() for x in group.get('labels') or []]
            for p in group.get('athletes') or []:
                athlete=p.get('athlete') or {}
                pid=str(athlete.get('id') or '')
                if not pid: continue
                rawstats=p.get('stats') or []
                if isinstance(rawstats,list):
                    stats=dict(zip(labels,rawstats))
                else: stats=rawstats
                n=_box_numbers(stats)
                cols=['points','fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','reb','ast','stl','blk','tov','pf','plus_minus']
                db.execute('''INSERT OR REPLACE INTO player_games(game_id,player_id,team,player_name,starter,played,
                    minutes,points,fgm,fga,fg3m,fg3a,ftm,fta,oreb,dreb,reb,ast,stl,blk,tov,pf,plus_minus,raw_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (gid,pid,abbr,athlete.get('displayName') or athlete.get('fullName') or '',
                     int(bool(p.get('starter'))),int(bool(p.get('didPlay',True))),n['minutes'],
                     *[n[c] for c in cols],json.dumps(p)))
                counts['players']+=1
    # Some ESPN responses omit team box scores but provide player totals. Never
    # synthesize a supposedly official team boxscore from incomplete players.
    return counts,team_stats

def _parse_plays(db,gid,payload):
    plays=payload.get('plays') or []
    if isinstance(plays,dict): plays=plays.get('items') or []
    counts={'plays':0,'shots':0,'substitutions':0,'fouls':0,'turnovers':0}
    for ix,p in enumerate(plays):
        typ=p.get('type') or {}
        if not isinstance(typ,dict): typ={'text':str(typ)}
        kind=str(typ.get('text') or typ.get('name') or typ.get('id') or '')
        desc=str(p.get('text') or p.get('description') or '')
        eid=str(p.get('id') or f'index-{ix}')
        per=p.get('period') or {}
        period=integer(per.get('number') if isinstance(per,dict) else per)
        tm=p.get('team') or {}
        abbr=tm.get('abbreviation') if isinstance(tm,dict) else ''
        athletes=p.get('athletes') or []
        first=athletes[0] if athletes else {}
        athlete=(first.get('athlete') or {}) if isinstance(first,dict) else {}
        pid=str(athlete.get('id') or '')
        coord=p.get('coordinate') or {}
        x=number(coord.get('x')) if isinstance(coord,dict) else None
        y=number(coord.get('y')) if isinstance(coord,dict) else None
        clock= p.get('clock') or {}
        clk=str(clock.get('displayValue') or '') if isinstance(clock,dict) else str(clock)
        seconds=clock_secs(clock)
        score=integer(p.get('scoreValue'),0)
        scoring=int(bool(p.get('scoringPlay')))
        shooting=int(bool(p.get('shootingPlay')))
        db.execute('''INSERT OR REPLACE INTO plays VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
          (gid,eid,ix,period,clk,seconds,abbr,pid,kind,desc,scoring,score,
           integer(p.get('homeScore')),integer(p.get('awayScore')),shooting,x,y,json.dumps(p)))
        counts['plays']+=1
        label=(kind+' '+desc).lower()
        if 'foul' in label: counts['fouls']+=1
        if 'turnover' in label: counts['turnovers']+=1
        if 'substitution' in label or 'enters the game' in label or 'sub in' in label:
            db.execute('INSERT OR REPLACE INTO substitutions VALUES(?,?,?,?,?,?,?,?)',
                       (gid,eid,period,seconds,abbr,pid,desc,json.dumps(p)))
            counts['substitutions']+=1
        # Shot attempts exclude free throws. Missing coordinates stay NULL.
        if shooting or ('jumper' in label or 'layup' in label or 'dunk' in label or 'hook shot' in label):
            if 'free throw' in label: continue
            made=int(scoring and score in (2,3))
            dist=number(p.get('shotDistance'))
            db.execute('INSERT OR REPLACE INTO shots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (gid,eid,period,seconds,abbr,pid,made,score if made else 0,x,y,dist,kind))
            counts['shots']+=1
    return counts

def _event_features(db,gid):
    rows=db.execute('SELECT * FROM plays WHERE game_id=? ORDER BY sequence',(gid,)).fetchall()
    for r in rows:
        label=(str(r['event_type'])+' '+str(r['description'])).lower()
        if 'free throw' in label: category='free_throw'
        elif r['shooting'] or any(x in label for x in ('jumper','layup','dunk','hook shot')): category='shot'
        elif 'foul' in label: category='foul'
        elif 'turnover' in label: category='turnover'
        elif 'rebound' in label: category='rebound'
        elif 'substitution' in label or 'enters the game' in label: category='substitution'
        elif 'timeout' in label: category='timeout'
        elif 'jump ball' in label: category='jump_ball'
        elif 'violation' in label: category='violation'
        else: category='other'
        subtype=r['event_type']
        period=r['period'];remaining=r['clock_seconds']
        elapsed=None
        if period and remaining is not None:
            elapsed=(period-1)*720+(720-remaining) if period<=4 else 2880+(period-5)*300+(300-remaining)
        margin=(r['home_score']-r['away_score']) if r['home_score'] is not None and r['away_score'] is not None else None
        clutch=int(period is not None and period>=4 and remaining is not None and remaining<=300 and margin is not None and abs(margin)<=5)
        shot=db.execute('SELECT distance_ft,points FROM shots WHERE game_id=? AND play_id=?',(gid,r['play_id'])).fetchone()
        dist=shot['distance_ft'] if shot else None
        is_three=None
        zone=None
        if category=='shot':
            if dist is not None:
                zone='rim' if dist<5 else 'paint' if dist<10 else 'midrange' if dist<22 else 'three'
                is_three=int(dist>=23)
            elif '3-pt' in label or 'three point' in label or '3pt' in label:
                zone='three';is_three=1
            elif 'layup' in label or 'dunk' in label:
                zone='rim';is_three=0
        db.execute('INSERT OR REPLACE INTO play_features VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (gid,r['play_id'],category,subtype,elapsed,margin,clutch,zone,is_three,
                    'event text; shot distance when available'))

def _team_game_context(db,gid,day,home,away):
    rows={r['team']:r for r in db.execute('SELECT * FROM team_games WHERE game_id=?',(gid,))}
    for team in (home,away):
        opponent=away if team==home else home
        o=rows.get(opponent)
        prev=db.execute("""SELECT MAX(g.game_date) FROM team_games t JOIN games g USING(game_id)
                 WHERE t.team=? AND g.game_date<? AND g.state='post' """,(team,str(day))).fetchone()[0]
        rest=(day-date.fromisoformat(prev)).days-1 if prev else None
        db.execute('INSERT OR REPLACE INTO game_team_context VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (gid,team,rest,int(rest==0) if rest is not None else None,
                    o['efg_pct'] if o else None,o['three_rate'] if o else None,
                    o['ft_rate'] if o else None,o['tov_pct'] if o else None,
                    o['oreb_pct'] if o else None,o['points'] if o else None))

def _period_scores(db,gid,payload):
    for t in (payload.get('header') or {}).get('competitions',[{}])[0].get('competitors',[]):
        abbr=_team_abbr(t)
        for line in t.get('linescores') or []:
            per=integer(line.get('period') or line.get('periodNumber'))
            if per is None: continue
            db.execute('INSERT OR REPLACE INTO period_scores VALUES(?,?,?,?)',
                       (gid,abbr,per,integer(line.get('value'),0)))

def _derive(db,gid,home,away,periods):
    """Possessions, pace and four factors from verified box scores, not invented PBP."""
    rows=db.execute('SELECT * FROM team_games WHERE game_id=?',(gid,)).fetchall()
    if len(rows)!=2: return
    by={r['team']:r for r in rows}
    if home not in by or away not in by: return
    length=48+max(0,(periods or 4)-4)*5
    poss={}
    for t,r in by.items():
        needed=[r[k] for k in ('fga','oreb','tov','fta')]
        poss[t]=max(0,r['fga']-r['oreb']+r['tov']+.44*r['fta']) if all(x is not None for x in needed) else None
    pace=(poss[home]+poss[away])/2*48/length if all(x is not None for x in poss.values()) else None
    for t,r in by.items():
        opp=away if t==home else home
        o=by[opp]
        p=poss[t]; q=poss[opp]
        def div(a,b): return a/b if a is not None and b is not None and b>0 else None
        off=div(r['points']*100,p)
        deff=div(o['points']*100,q)
        efg=div((r['fgm'] or 0)+.5*(r['fg3m'] or 0),r['fga']) if r['fgm'] is not None and r['fg3m'] is not None else None
        ts=div(r['points'],2*(r['fga']+.44*r['fta'])) if r['points'] is not None and r['fga'] is not None and r['fta'] is not None else None
        tov=div(r['tov'],p)
        oreb=div(r['oreb'],r['oreb']+o['dreb']) if r['oreb'] is not None and o['dreb'] is not None else None
        ftr=div(r['fta'],r['fga']); three=div(r['fg3a'],r['fga'])
        db.execute('''UPDATE team_games SET minutes=?,possessions_est=?,pace_est=?,off_rating_est=?,
             def_rating_est=?,net_rating_est=?,efg_pct=?,ts_pct=?,tov_pct=?,oreb_pct=?,
             ft_rate=?,three_rate=? WHERE game_id=? AND team=?''',
             (length,p,pace,off,deff,off-deff if off is not None and deff is not None else None,
              efg,ts,tov,oreb,ftr,three,gid,t))

def ingest_summary(db,g,day,payload):
    gid=str(g['id']); home=g['home']['abbr']; away=g['away']['abbr']
    with db:
        _upsert_game(db,g,day)
        _store_raw(db,gid,'summary',payload)
        counts,team_stats=_parse_summary_box(db,gid,payload,home,away)
        play_counts=_parse_plays(db,gid,payload)
        _period_scores(db,gid,payload)
        header=payload.get('header') or {}
        competitions=header.get('competitions') or [{}]
        comp=competitions[0] if competitions else {}
        status=comp.get('status') or {}
        periods=integer(status.get('period') or status.get('periods'),4)
        if not periods:
            periods=max([integer(x.get('period'),4) for x in (comp.get('competitors') or [])] or [4])
        venue=comp.get('venue') or {}
        db.execute('''UPDATE games SET periods=?,venue=?,attendance=?,detail_status=?,detail_error=NULL WHERE game_id=?''',
                   (periods,venue.get('fullName'),integer(comp.get('attendance')),
                    'complete' if counts['teams']==2 and counts['players']>0 else 'partial',gid))
        _derive(db,gid,home,away,periods)
        _event_features(db,gid)
        from pbp_learning import index_event_video_metadata
        index_event_video_metadata(db,gid)
        _team_game_context(db,gid,day,home,away)
        cover={
          'team_boxscore':(counts['teams']==2,'Official per-game fields where supplied'),
          'player_boxscore':(counts['players']>0,'Player stats, minutes and personal fouls'),
          'play_by_play':(play_counts['plays']>0,'All provider events and source JSON retained'),
          'shot_locations':(db.execute('SELECT COUNT(*) FROM shots WHERE game_id=? AND x IS NOT NULL',(gid,)).fetchone()[0]>0,
                             'Coordinates only when provider supplies them'),
          'substitutions':(play_counts['substitutions']>0,'Raw substitutions; not guaranteed complete'),
          'individual_defender_tracking':(False,'Not in public ESPN summary; no fabricated matchup/contest data'),
          'verified_lineup_stints':(False,'Starter/substitution completeness not guaranteed; not reconstructed'),
          'true_possession_boundaries':(False,'Only boxscore-estimated possessions available; no false event segmentation'),
          'foul_events':(play_counts['fouls']>0,'Event-level fouls preserved from provider descriptions'),
          'clutch_events':(play_counts['plays']>0,'Clock/score context derived when provider gives both'),
          'possessions':(all(db.execute('SELECT possessions_est FROM team_games WHERE game_id=? AND team=?',(gid,t)).fetchone() and
                             db.execute('SELECT possessions_est FROM team_games WHERE game_id=? AND team=?',(gid,t)).fetchone()[0] is not None for t in (home,away)),
                         'Estimated FGA - OREB + TOV + 0.44*FTA; not exact possessions'),
          'pace':((db.execute('SELECT pace_est FROM team_games WHERE game_id=? LIMIT 1',(gid,)).fetchone() or [None])[0] is not None,
                  'Estimated possessions per 48 minutes; overtime adjusted'),
        }
        for category,(available,note) in cover.items():
            db.execute('INSERT OR REPLACE INTO coverage VALUES(?,?,?,?)',(gid,category,int(bool(available)),note))
    return {**counts,**play_counts,'game_id':gid}

def sync_day(db,day: date,include_injuries=False,throttle=0.15):
    """Discover every scheduled game; only ingest details after games are final."""
    games=schedule(day)
    with db:
        for g in games: _upsert_game(db,g,day)
        db.execute('INSERT OR REPLACE INTO sync_state VALUES(?,?)',('last_schedule_day',str(day)))
    result={'day':str(day),'scheduled':len(games),'ingested':0,'skipped':0,'errors':[]}
    for g in games:
        gid=str(g['id'])
        if g.get('state')!='post':
            result['skipped']+=1;continue
        status=db.execute('SELECT detail_status FROM games WHERE game_id=?',(gid,)).fetchone()
        if status and status[0]=='complete':
            result['skipped']+=1;continue
        try:
            payload=get_json(f'{ESPN}/summary',{'event':gid})
            ingest_summary(db,g,day,payload)
            result['ingested']+=1
            with db: db.execute('DELETE FROM sync_errors WHERE day=? AND game_id=?',(str(day),gid))
        except Exception as exc:
            error=f'{type(exc).__name__}: {str(exc)[:200]}'
            result['errors'].append(f'{gid}: {error}')
            with db:
                db.execute('UPDATE games SET detail_status=?,detail_error=? WHERE game_id=?',('error',error,gid))
                db.execute('INSERT OR REPLACE INTO sync_errors VALUES(?,?,?,?)',(str(day),gid,error,utc()))
        if throttle: time.sleep(throttle)
    if include_injuries and day>=date.today():
        ids={str(g[side].get('id','')) for g in games for side in ('home','away')}
        for team_id in ids- {''}:
            try:
                entries,_=injuries(team_id)
                with db:
                    for item in entries:
                        db.execute('INSERT OR REPLACE INTO injury_snapshots VALUES(?,?,?,?,?,?)',
                                   (str(day),team_id,item['player'],item['status'],item.get('note',''),utc()))
            except Exception as exc:
                result['errors'].append(f'injuries {team_id}: {type(exc).__name__}')
    with db: db.execute('INSERT OR REPLACE INTO sync_state VALUES(?,?)',('last_sync_utc',utc()))
    return result

def sync_recent(days=3,db_path=DB_PATH):
    with connect(db_path) as db:
        return [sync_day(db,date.today()-timedelta(days=n),include_injuries=(n==0))
                for n in range(days-1,-1,-1)]

def backfill_batch(db, start: date, end: date, max_days=7,throttle=.15):
    """Persistent cursor advances only when the day was successfully discovered."""
    cursor=db.execute('SELECT value FROM sync_state WHERE key=?',('backfill_cursor',)).fetchone()
    current=max(start,date.fromisoformat(cursor[0])) if cursor else start
    result=[]
    while current<=end and len(result)<max_days:
        r=sync_day(db,current,throttle=throttle)
        result.append(r)
        with db:
            db.execute('INSERT OR REPLACE INTO sync_state VALUES(?,?)',('backfill_cursor',str(current+timedelta(days=1))))
        current+=timedelta(days=1)
    return result

def overview(db):
    def scalar(q):return db.execute(q).fetchone()[0]
    out={
      'games_discovered':scalar('SELECT COUNT(*) FROM games'),
      'final_games':scalar("SELECT COUNT(*) FROM games WHERE state='post'"),
      'games_complete':scalar("SELECT COUNT(*) FROM games WHERE detail_status='complete'"),
      'games_partial':scalar("SELECT COUNT(*) FROM games WHERE detail_status='partial'"),
      'games_failed':scalar("SELECT COUNT(*) FROM games WHERE detail_status='error'"),
      'players_logged':scalar('SELECT COUNT(*) FROM player_games'),
      'team_games_logged':scalar('SELECT COUNT(*) FROM team_games'),
      'plays_logged':scalar('SELECT COUNT(*) FROM plays'),
      'shots_logged':scalar('SELECT COUNT(*) FROM shots'),
      'shots_with_coordinates':scalar('SELECT COUNT(*) FROM shots WHERE x IS NOT NULL'),
      'substitutions_logged':scalar('SELECT COUNT(*) FROM substitutions'),
      'injury_snapshots':scalar('SELECT COUNT(*) FROM injury_snapshots'),
      'event_features':scalar('SELECT COUNT(*) FROM play_features'),
      'game_team_context':scalar('SELECT COUNT(*) FROM game_team_context'),
      'estimated_pace_games':scalar('SELECT COUNT(DISTINCT game_id) FROM team_games WHERE pace_est IS NOT NULL'),
      'last_game_date':scalar('SELECT MAX(game_date) FROM games'),
      'first_game_date':scalar('SELECT MIN(game_date) FROM games'),
      'backfill_cursor':(db.execute('SELECT value FROM sync_state WHERE key=?',('backfill_cursor',)).fetchone() or [None])[0],
      'last_sync_utc':(db.execute('SELECT value FROM sync_state WHERE key=?',('last_sync_utc',)).fetchone() or [None])[0],
    }
    return out

def historical_context(db, team: str, player_name: str, cutoff: str, lookback=20):
    """Strict as-of cutoff; excludes the prediction date and all later games."""
    rows=db.execute('''SELECT pg.minutes,pg.pf,pg.points,g.game_date,pg.starter
        FROM player_games pg JOIN games g USING(game_id)
        WHERE pg.team=? AND lower(pg.player_name)=lower(?) AND g.game_date<?
          AND g.state='post' AND pg.played=1 AND pg.minutes>0
        ORDER BY g.game_date DESC LIMIT ?''',(team,player_name,cutoff,lookback)).fetchall()
    if not rows:return None
    import statistics
    return {'games':len(rows),'minutes_mean':statistics.mean(r['minutes'] for r in rows),
            'minutes_sd':statistics.pstdev(r['minutes'] for r in rows) if len(rows)>1 else None,
            'fouls_per_36':statistics.mean((r['pf'] or 0)*36/r['minutes'] for r in rows),
            'starter_rate':statistics.mean(r['starter'] or 0 for r in rows),
            'last_game':rows[0]['game_date']}

def team_context(db,team: str,cutoff: str,lookback=20):
    rows=db.execute('''SELECT tg.pace_est,tg.off_rating_est,tg.def_rating_est,
        tg.oreb_pct,tg.efg_pct,tg.tov_pct,tg.ft_rate,g.game_date
        FROM team_games tg JOIN games g USING(game_id)
        WHERE tg.team=? AND g.game_date<? AND g.state='post'
        ORDER BY g.game_date DESC LIMIT ?''',(team,cutoff,lookback)).fetchall()
    if not rows:return None
    import statistics
    out={'games':len(rows),'last_game':rows[0]['game_date']}
    for k in ('pace_est','off_rating_est','def_rating_est','oreb_pct','efg_pct','tov_pct','ft_rate'):
        vals=[r[k] for r in rows if r[k] is not None]
        out[k]=statistics.mean(vals) if vals else None
        out[k+'_samples']=len(vals)
    return out

def apply_archive_priors(teams,players,cutoff,db_path=DB_PATH):
    """Conservative as-of archive correction; does not double-count season evidence.

    NBA.com season + prior-season estimates remain primary. Archive only provides
    modest extra information about recent minutes and pace. Never change roster
    membership, eligibility, shot skill or defensive ratings without evidence.
    """
    import numpy as np
    teams=teams.copy();players=players.copy()
    teams['ARCHIVE_PACE_GAMES']=0
    players['ARCHIVE_MINUTES_GAMES']=0
    players['ARCHIVE_FOULS_PER_36']=np.nan
    players['PBP_SHOT_PROFILE_GAMES']=0
    if not Path(db_path).exists():return teams,players,{'status':'empty','players':0,'teams':0}
    matched=0;tm=0;pbp_profiles=0
    from pbp_learning import player_shot_profile
    with connect(db_path) as db:
        for ix,row in players.iterrows():
            c=historical_context(db,str(row['TEAM_ABBREVIATION']),str(row['PLAYER_NAME']),cutoff)
            if not c:continue
            matched+=1
            players.at[ix,'ARCHIVE_MINUTES_GAMES']=c['games']
            # As-of play-by-play shot selection is allowed only after strict
            # shot-to-boxscore coverage checks. Never use raw video heuristics.
            if 'FG3A' in players and 'FGA' in players:
                pbp=player_shot_profile(db,str(row['TEAM_ABBREVIATION']),
                                        str(row['PLAYER_NAME']),cutoff)
                if pbp:
                    fga=number(row.get('FGA'));fg3a=number(row.get('FG3A'))
                    if fga is not None and fga>0 and fg3a is not None:
                        old_rate=max(0,min(1,fg3a/fga))
                        # Bounded 12% contribution; season NBA.com remains primary.
                        weight=min(.12,.12*pbp['games']/(pbp['games']+10))
                        new_rate=(1-weight)*old_rate+weight*pbp['three_attempt_rate']
                        new_fg3a=min(fga,max(0,fga*new_rate))
                        if 'FG3M' in players and fg3a>0:
                            fg3m=number(row.get('FG3M'))
                            if fg3m is not None:
                                players.at[ix,'FG3M']=min(new_fg3a,max(0,fg3m*new_fg3a/fg3a))
                        players.at[ix,'FG3A']=new_fg3a
                        players.at[ix,'PBP_SHOT_PROFILE_GAMES']=pbp['games']
                        pbp_profiles+=1
            # One game's fouls are noisy: shrink the archive rate toward
            # a generic 2.5 fouls / 36 before it enters the coupled engine.
            foul_weight=c['games']/(c['games']+12.)
            players.at[ix,'ARCHIVE_FOULS_PER_36']=(1-foul_weight)*2.5+foul_weight*c['fouls_per_36']
            if 'PF' in players and c['games']>=3:
                base=number(row.get('PF'))
                if base is not None:
                    w=min(.2,.2*c['games']/(c['games']+10.))
                    players.at[ix,'PF']=(1-w)*base+w*players.at[ix,'ARCHIVE_FOULS_PER_36']*max(1,float(row['MIN']))/36.
            # Historical NBA.com already blends last season; local database is
            # an incremental *role* signal, capped to 25% to avoid double counting.
            if c['games']>=3 and c['minutes_mean'] is not None:
                w=min(.25,.25*c['games']/(c['games']+8))
                players.at[ix,'MIN']=max(0,min(42,(1-w)*float(row['MIN'])+w*c['minutes_mean']))
        for ix,row in teams.iterrows():
            c=team_context(db,str(row['TEAM_ABBREVIATION']),cutoff)
            if not c or c['pace_est'] is None:continue
            tm+=1
            teams.at[ix,'ARCHIVE_PACE_GAMES']=c['pace_est_samples']
            if c['pace_est_samples']>=3:
                w=min(.20,.20*c['pace_est_samples']/(c['pace_est_samples']+10))
                teams.at[ix,'PACE']=(1-w)*float(row['PACE'])+w*c['pace_est']
    return teams,players,{'status':'ok','players':matched,'teams':tm,
                          'pbp_shot_profiles':pbp_profiles,
                          'note':'As-of prior games only; minutes, pace, and validated PBP shot selection are lightly blended.'}

def cli(argv=None):
    parser=argparse.ArgumentParser(description='COURTSIDE IQ complete-game NBA archive')
    parser.add_argument('--recent-days',type=int,default=3)
    parser.add_argument('--backfill-start',default='2023-10-01')
    parser.add_argument('--max-days',type=int,default=7)
    parser.add_argument('--no-backfill',action='store_true')
    parser.add_argument('--db',type=Path,default=DB_PATH)
    args=parser.parse_args(argv)
    with connect(args.db) as db:
        results=[]
        for n in range(max(1,min(30,args.recent_days))-1,-1,-1):
            day=date.today()-timedelta(days=n)
            try: results.append(sync_day(db,day,include_injuries=(n==0)))
            except Exception as exc: results.append({'day':str(day),'error':str(exc)})
        if not args.no_backfill:
            try: results+=backfill_batch(db,date.fromisoformat(args.backfill_start),
                         date.today()-timedelta(days=1),max_days=max(0,min(args.max_days,100)))
            except Exception as exc: results.append({'backfill_error':str(exc)})
        print(json.dumps({'results':results,'overview':overview(db)},indent=2))

if __name__=='__main__':cli()
