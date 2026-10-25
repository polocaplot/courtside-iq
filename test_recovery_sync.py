from datetime import date
from pathlib import Path
import json
from nba_archive import connect
from recovery_sync import run, candidate_days


def test_recovers_missed_days_and_retries_failed_details(tmp_path):
    db_path = tmp_path/'history.sqlite3'
    root = tmp_path/'settings'
    root.mkdir()
    (root/'recovery_state.json').write_text(json.dumps({'next_day':'2026-10-19'}))
    with connect(db_path) as db:
        db.execute("INSERT INTO games(game_id,game_date,state,detail_status,updated_utc) VALUES('old','2026-10-18','post','error','now')")
        db.execute("INSERT INTO sync_errors(day,game_id,error,checked_utc) VALUES('2026-10-18','old','failed','now')")
        db.commit()
    seen=[]
    def fake_sync(db, day, include_injuries=False):
        seen.append(str(day))
        if str(day)=='2026-10-20':
            raise ConnectionError('offline')
        if str(day)=='2026-10-18':
            db.execute("UPDATE games SET detail_status='complete' WHERE game_id='old'")
            db.execute("DELETE FROM sync_errors WHERE game_id='old'")
            db.commit()
        return {'errors':[]}
    result=run(db_path=db_path,root=root,today=date(2026,10,22),start=date(2026,10,18),max_days=5,recent_days=2,sync_fn=fake_sync)
    assert '2026-10-18' in seen
    assert '2026-10-22' in seen
    assert result['next_catchup_day']=='2026-10-20'
    assert result['missing_details']==0
    assert json.loads((root/'recovery_report.json').read_text())['sqlite_integrity']=='ok'
    seen.clear()
    def succeeding(db,day,include_injuries=False):
        seen.append(str(day))
        return {'errors':[]}
    second=run(db_path=db_path,root=root,today=date(2026,10,22),start=date(2026,10,18),max_days=5,recent_days=2,sync_fn=succeeding)
    assert second['next_catchup_day']=='2026-10-23'


def test_no_duplicate_date_attempts(tmp_path):
    db_path=tmp_path/'db.sqlite3'
    with connect(db_path) as db:
        days, historic=candidate_days(db,date(2026,10,22),{'next_day':'2026-10-20'},date(2026,10,20),5,4)
    assert len(days)==len(set(days))
    assert historic==[date(2026,10,20),date(2026,10,21),date(2026,10,22)]
