"""Crash-safe catch-up scheduler for the COURTSIDE IQ archive.

Always uses ~/.courtsideiq, never a database inside a replaceable app bundle.
Daily cursors only advance after successful schedule retrieval. Failed game details
remain eligible for retry even when the date cursor has moved forward.
"""
from __future__ import annotations
import argparse, fcntl, json, os, sqlite3, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from nba_archive import connect, sync_day, DB_PATH

ROOT = Path.home() / '.courtsideiq'
STATE = ROOT / 'recovery_state.json'
REPORT = ROOT / 'recovery_report.json'

def stamp(): return datetime.now(timezone.utc).isoformat(timespec='seconds')

def read_state(path=STATE):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {}

def atomic_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp,path)

def candidate_days(db, today, state, start, max_days, recent_days):
    """Retry failed/incomplete games, then cover every missed calendar day in order."""
    pending = set()
    # Retry incomplete summaries and recorded fetch errors regardless of age.
    for row in db.execute("SELECT DISTINCT game_date FROM games WHERE state='post' AND detail_status != 'complete'"):
        if row[0]: pending.add(date.fromisoformat(row[0]))
    for row in db.execute('SELECT DISTINCT day FROM sync_errors'):
        if row[0]: pending.add(date.fromisoformat(row[0]))
    # Last few days are revisited to pick up late-final games and corrected feeds.
    for offset in range(max(1,recent_days)):
        pending.add(today-timedelta(days=offset))
    cursor = date.fromisoformat(state.get('next_day', str(start)))
    cursor = max(start, cursor)
    # Prioritize old gaps so long absences are eventually filled, without starving recent games.
    historic=[]
    while cursor<=today and len(historic)<max_days:
        historic.append(cursor)
        cursor+=timedelta(days=1)
    retry=sorted(d for d in pending if start<=d<=today)
    recent=set(today-timedelta(days=i) for i in range(max(1,recent_days)))
    # Recent days first, followed by old retries and the catch-up cursor.
    days=sorted(recent, reverse=True)+[d for d in retry if d not in recent]
    days += [d for d in historic if d not in days]
    return days, historic

def backup_database(db, root, now=None):
    """Consistent SQLite backup, including WAL data; retain the newest three."""
    now=now or datetime.now(timezone.utc)
    directory=Path(root)/'database_backups'
    directory.mkdir(parents=True,exist_ok=True)
    latest=sorted(directory.glob('nba_archive_*.sqlite3'))
    if latest and (now-datetime.fromtimestamp(latest[-1].stat().st_mtime,timezone.utc)).total_seconds()<86400:
        return str(latest[-1])
    dest=directory/('nba_archive_'+now.strftime('%Y%m%d_%H%M%S')+'.sqlite3')
    with sqlite3.connect(str(dest)) as out:
        db.backup(out)
        if out.execute('PRAGMA quick_check').fetchone()[0]!='ok':
            dest.unlink(missing_ok=True)
            raise RuntimeError('Backup failed integrity check')
    for old in sorted(directory.glob('nba_archive_*.sqlite3'))[:-3]:
        old.unlink(missing_ok=True)
    return str(dest)

def run(db_path=DB_PATH, root=ROOT, today=None, start=None, max_days=30, recent_days=4,
        sync_fn=sync_day):
    today=today or date.today()
    start=start or date(2023,10,1)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    state_file=root/'recovery_state.json'
    report_file=root/'recovery_report.json'
    state=read_state(state_file)
    summary={'started_utc':stamp(),'database':str(db_path),'days_attempted':[],
             'days_succeeded':[], 'errors':[], 'missing_details':0}
    with connect(db_path) as db:
        check=db.execute('PRAGMA quick_check').fetchone()[0]
        summary['sqlite_integrity']=check
        if check!='ok':
            summary['errors'].append('Database integrity check failed; refusing to write')
            atomic_json(report_file,summary)
            raise RuntimeError('SQLite integrity check failed: '+str(check))
        days, historic=candidate_days(db,today,state,start,max_days,recent_days)
        # Track each contiguous cursor day only if its schedule fetched successfully.
        cursor=date.fromisoformat(state.get('next_day',str(start)))
        cursor=max(start,cursor)
        successes=set()
        for day in days:
            summary['days_attempted'].append(str(day))
            try:
                result=sync_fn(db,day,include_injuries=False)
                # Detail fetch errors do not block schedule cursor: retry through sync_errors.
                successes.add(day)
                summary['days_succeeded'].append(str(day))
                summary['errors'] += [{'day':str(day),'error':e} for e in result.get('errors',[])]
            except Exception as exc:
                summary['errors'].append({'day':str(day),'error':f'{type(exc).__name__}: {exc}'[:240]})
        while cursor in successes:
            cursor+=timedelta(days=1)
        # Advance through already covered dates before this run (if a later retry fails).
        # Cursor only ever increases over contiguous successful dates in this run.
        state.update({'next_day':str(cursor),'last_attempt_utc':stamp(),
                      'last_success_utc':stamp() if successes else state.get('last_success_utc'),
                      'last_error_count':len(summary['errors'])})
        summary['next_catchup_day']=str(cursor)
        summary['missing_details']=db.execute("SELECT count(*) FROM games WHERE state='post' AND detail_status != 'complete'").fetchone()[0]
        summary['recorded_errors']=db.execute('SELECT count(*) FROM sync_errors').fetchone()[0]
        summary['games']=db.execute('SELECT count(*) FROM games').fetchone()[0]
        summary['plays']=db.execute('SELECT count(*) FROM plays').fetchone()[0]
        try:
            summary['latest_backup']=backup_database(db,root)
        except Exception as exc:
            summary['errors'].append({'backup_error':str(exc)[:240]})
    summary['finished_utc']=stamp()
    atomic_json(state_file,state)
    atomic_json(report_file,summary)
    return summary

def cli(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-days',type=int,default=30)
    parser.add_argument('--recent-days',type=int,default=4)
    parser.add_argument('--start',default='2023-10-01')
    parser.add_argument('--db',type=Path,default=DB_PATH)
    args=parser.parse_args(argv)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'archive_sync.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another archive sync is running; skipping duplicate worker.')
            return 0
        report=run(db_path=args.db,start=date.fromisoformat(args.start),
                   max_days=max(1,min(args.max_days,365)),recent_days=max(1,min(args.recent_days,14)))
        print(json.dumps(report,indent=2))
        return 0 if not report['errors'] else 2

if __name__=='__main__': sys.exit(cli())
