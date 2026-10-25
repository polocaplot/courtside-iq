import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import date
import pytest
from nba_archive import (connect, ingest_summary, overview, historical_context,
                         team_context, apply_archive_priors, minutes, clock_secs, backfill_batch)

GAME={'id':'999','home':{'abbr':'ATL','score':'115'},'away':{'abbr':'HOU','score':'110'},
      'tipoff':'2025-01-01T00:00:00Z','state':'post','status':'STATUS_FINAL'}

def team(abbr,pts):
    return {'team':{'abbreviation':abbr},'statistics':[
      {'name':'points','displayValue':str(pts)},
      {'name':'fieldGoalsMade-fieldGoalsAttempted','displayValue':'40-85'},
      {'name':'threePointFieldGoalsMade-threePointFieldGoalsAttempted','displayValue':'12-33'},
      {'name':'freeThrowsMade-freeThrowsAttempted','displayValue':'23-29'},
      {'name':'offensiveRebounds','displayValue':'11'},
      {'name':'defensiveRebounds','displayValue':'32'},
      {'name':'turnovers','displayValue':'13'},
      {'name':'assists','displayValue':'24'},
      {'name':'fouls','displayValue':'18'}]}

def players(abbr,pid,name):
    return {'team':{'abbreviation':abbr},'statistics':[{'labels':['MIN','PTS','FG','3PT','FT','OREB','DREB','REB','AST','STL','BLK','TO','PF','+/-'],
       'athletes':[{'athlete':{'id':pid,'displayName':name},'starter':True,'didPlay':True,
       'stats':['35:30','25','9-18','3-7','4-5','2','5','7','6','1','1','2','3','+5']}]}]}

PAYLOAD={'boxscore':{'teams':[team('ATL',115),team('HOU',110)],
                     'players':[players('ATL','123','Jalen Johnson'),players('HOU','456','Amen Thompson')]},
         'header':{'competitions':[{'status':{'period':4},'venue':{'fullName':'Arena'},'attendance':16000,
                    'competitors':[{'team':{'abbreviation':'ATL'},'linescores':[{'period':1,'value':27}]},
                                   {'team':{'abbreviation':'HOU'},'linescores':[{'period':1,'value':25}]}]}]},
         'plays':[{'id':'a','type':{'text':'Made Layup'},'text':'Jalen Johnson makes layup',
                   'period':{'number':1},'clock':{'displayValue':'10:22'},
                   'team':{'abbreviation':'ATL'},'athletes':[{'athlete':{'id':'123'}}],
                   'scoringPlay':True,'scoreValue':2,'shootingPlay':True,
                   'homeScore':2,'awayScore':0,'coordinate':{'x':22,'y':11}},
                  {'id':'b','type':{'text':'Personal Foul'},'text':'Amen Thompson foul',
                   'period':{'number':1},'clock':{'displayValue':'10:21'},'team':{'abbreviation':'HOU'}}]}

def test_ingest_idempotent_and_pace(tmp_path):
    with connect(tmp_path/'archive.sqlite') as db:
        ingest_summary(db,GAME,date(2025,1,1),PAYLOAD)
        ingest_summary(db,GAME,date(2025,1,1),PAYLOAD)
        s=overview(db)
        assert s['games_discovered']==1
        assert s['team_games_logged']==2
        assert s['players_logged']==2
        assert s['plays_logged']==2
        assert s['shots_logged']==1
        assert s['estimated_pace_games']==1
        assert s['shots_with_coordinates']==1
        assert db.execute('SELECT minutes,pf,fg3a FROM player_games WHERE player_id="123"').fetchone()[:]==(35.5,3,7)
        assert db.execute('SELECT pace_est,possessions_est FROM team_games WHERE team="ATL"').fetchone()[0]>80
        assert historical_context(db,'ATL','Jalen Johnson','2025-01-02')['minutes_mean']==35.5
        assert historical_context(db,'ATL','Jalen Johnson','2025-01-01') is None
        assert team_context(db,'ATL','2025-01-02')['pace_est'] is not None
        assert db.execute('SELECT available FROM coverage WHERE category="individual_defender_tracking"').fetchone()[0]==0

def test_archive_priors_asof(tmp_path):
    import pandas as pd
    path=tmp_path/'db.sqlite'
    with connect(path) as db:ingest_summary(db,GAME,date(2025,1,1),PAYLOAD)
    teams=pd.DataFrame([{'TEAM_ABBREVIATION':'ATL','PACE':100.}])
    players=pd.DataFrame([{'TEAM_ABBREVIATION':'ATL','PLAYER_NAME':'Jalen Johnson','MIN':31.}])
    a,b,info=apply_archive_priors(teams,players,'2025-01-02',path)
    assert info['players']==1
    assert b['ARCHIVE_FOULS_PER_36'].iloc[0]>0
    a,b,info=apply_archive_priors(teams,players,'2025-01-01',path)
    assert info['players']==0
    assert b['MIN'].iloc[0]==31

def test_time_parsing():
    assert minutes('31:30')==31.5
    assert clock_secs({'displayValue':'8:30'})==510
