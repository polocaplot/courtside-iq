"""Optional automatic video discovery and processing; no stream ripping or DRM bypass.

YouTube Data API searches return metadata only. Video analysis uses only locally
supplied footage the user has permission to process. Never treats event clips
as complete-game optical tracking.
"""
from __future__ import annotations
import json, os, re, sqlite3, urllib.parse, urllib.request
from datetime import date, timedelta
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path.home()/'.courtsideiq'
CLIPS=ROOT/'event_clips'

# Deliberately conservative: only clear two-team, date-aligned matches are verified.
TEAM_NAMES={
 'ATL':('atlanta hawks','hawks','atlanta','atl'), 'BOS':('boston celtics','celtics','boston','bos'),
 'BKN':('brooklyn nets','nets','brooklyn','bkn'), 'BRK':('brooklyn nets','nets','brooklyn','bkn'),
 'CHA':('charlotte hornets','hornets','charlotte','cha'), 'CHI':('chicago bulls','bulls','chicago','chi'),
 'CLE':('cleveland cavaliers','cavaliers','cavs','cleveland','cle'),
 'DAL':('dallas mavericks','mavericks','mavs','dallas','dal'), 'DEN':('denver nuggets','nuggets','denver','den'),
 'DET':('detroit pistons','pistons','detroit','det'), 'GSW':('golden state warriors','warriors','golden state','gsw'),
 'HOU':('houston rockets','rockets','houston','hou'), 'IND':('indiana pacers','pacers','indiana','ind'),
 'LAC':('los angeles clippers','la clippers','clippers','lac'),
 'LAL':('los angeles lakers','la lakers','lakers','lal'),
 'MEM':('memphis grizzlies','grizzlies','memphis','mem'), 'MIA':('miami heat','heat','miami','mia'),
 'MIL':('milwaukee bucks','bucks','milwaukee','mil'), 'MIN':('minnesota timberwolves','timberwolves','wolves','minnesota','min'),
 'NOP':('new orleans pelicans','pelicans','new orleans','nop'), 'NO':('new orleans pelicans','pelicans','new orleans','nop'),
 'NYK':('new york knicks','knicks','new york','nyk'), 'OKC':('oklahoma city thunder','thunder','oklahoma city','okc'),
 'ORL':('orlando magic','magic','orlando','orl'), 'PHI':('philadelphia 76ers','76ers','sixers','philadelphia','phi'),
 'PHX':('phoenix suns','suns','phoenix','phx'), 'POR':('portland trail blazers','trail blazers','blazers','portland','por'),
 'SAC':('sacramento kings','kings','sacramento','sac'), 'SAS':('san antonio spurs','spurs','san antonio','sas'),
 'TOR':('toronto raptors','raptors','toronto','tor'), 'UTA':('utah jazz','jazz','utah','uta'),
 'WAS':('washington wizards','wizards','washington','was'), 'WSH':('washington wizards','wizards','washington','was')}


def setup(db):
    db.execute('''CREATE TABLE IF NOT EXISTS video_discoveries (
        game_id TEXT NOT NULL, provider TEXT NOT NULL, video_id TEXT NOT NULL,
        title TEXT, url TEXT, channel TEXT, published TEXT, checked_utc TEXT,
        status TEXT NOT NULL DEFAULT 'candidate_unverified',
        PRIMARY KEY(game_id,provider,video_id))''')
    cols={r[1] for r in db.execute('PRAGMA table_info(video_discoveries)')}
    for name,sql_type in [('verification_reason','TEXT'),('duration_seconds','INTEGER'),('video_kind','TEXT')]:
        if name not in cols:db.execute(f'ALTER TABLE video_discoveries ADD COLUMN {name} {sql_type}')


def _mentions(title,team):
    aliases=TEAM_NAMES.get(str(team).upper(),(str(team).lower(),))
    text=re.sub(r'[^a-z0-9]+',' ',title.lower()).strip()
    return any(re.search(r'(?<![a-z0-9])'+re.escape(alias)+r'(?![a-z0-9])',text) for alias in aliases)


def _explicit_dates(title):
    dates=set()
    for y,m,d in re.findall(r'\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b',title):
        try:dates.add(date(int(y),int(m),int(d)))
        except ValueError:pass
    months='january february march april may june july august september october november december'.split()
    for match in re.finditer(r'\b('+'|'.join(months)+r')\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2})\b',title,re.I):
        try:dates.add(date(int(match[3]),months.index(match[1].lower())+1,int(match[2])))
        except ValueError:pass
    for match in re.finditer(r'\b(\d{1,2})/([0-3]?\d)/(20\d{2})\b',title):
        try:dates.add(date(int(match[3]),int(match[1]),int(match[2])))
        except ValueError:pass
    return dates


def verify_candidate(video,away,home,game_date):
    """Return (status, reason); metadata verification is NOT footage validation."""
    title=video.get('title') or ''
    try:day=date.fromisoformat(str(game_date)[:10])
    except ValueError:return 'candidate_unverified','invalid_game_date'
    if re.search(r'\b(summer league|preseason|all.star|2k|simulation|reaction|prediction|preview)\b',title,re.I):
        return 'rejected_mismatch','non_regular_game_or_non_game_video'
    if not (_mentions(title,away) and _mentions(title,home)):
        return 'rejected_mismatch','both_teams_not_identified'
    if str(away).upper()==str(home).upper():return 'rejected_mismatch','invalid_matchup'
    mentioned=_explicit_dates(title)
    if mentioned and day not in mentioned:return 'rejected_mismatch','title_date_mismatch'
    published=video.get('published') or ''
    try:publish_day=date.fromisoformat(published[:10])
    except ValueError:return 'candidate_unverified','missing_publication_date'
    if not day <= publish_day <= day+timedelta(days=3):
        return 'rejected_mismatch','publication_date_outside_game_window'
    # Even a two-team, date-aligned title is only a metadata match.
    return 'verified_metadata','teams_and_date_match_metadata_only'


def _parse_duration(iso):
    m=re.fullmatch(r'P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',iso or '')
    if not m:return None
    return int(m[1] or 0)*3600+int(m[2] or 0)*60+int(m[3] or 0)


def _fetch_durations(video_ids,api_key,opener):
    if not video_ids:return {}
    params=urllib.parse.urlencode({'part':'contentDetails','id':','.join(video_ids[:50]),'key':api_key})
    req=urllib.request.Request('https://www.googleapis.com/youtube/v3/videos?'+params,
        headers={'User-Agent':'CourtsideIQ/1.0'})
    with opener(req,timeout=12) as response:payload=json.load(response)
    return {x['id']:_parse_duration((x.get('contentDetails') or {}).get('duration'))
            for x in payload.get('items',[]) if x.get('id')}


def search_youtube(game_id, away, home, game_date, api_key, opener=None):
    """Find metadata candidates; never download or assert full-game coverage."""
    if not api_key:return []
    q=f'{away} {home} NBA {game_date} game highlights'
    params=urllib.parse.urlencode({'part':'snippet','q':q,'type':'video',
        'maxResults':10,'key':api_key,'safeSearch':'moderate'})
    req=urllib.request.Request('https://www.googleapis.com/youtube/v3/search?'+params,
                               headers={'User-Agent':'CourtsideIQ/1.0'})
    open_fn=opener or urllib.request.urlopen
    with open_fn(req,timeout=12) as response:payload=json.load(response)
    found=[]
    for item in payload.get('items',[]):
        vid=item.get('id',{}).get('videoId')
        if not isinstance(vid,str) or not vid:continue
        snippet=item.get('snippet') or {}
        v={'game_id':str(game_id),'provider':'youtube',
            'video_id':vid,'title':snippet.get('title',''),
            'url':'https://www.youtube.com/watch?v='+urllib.parse.quote(vid),
            'channel':snippet.get('channelTitle',''),
            'published':snippet.get('publishedAt','')}
        v['status'],v['verification_reason']=verify_candidate(v,away,home,game_date)
        found.append(v)
    # Only spend an extra quota unit on video metadata for plausibly matched clips.
    matching=[v['video_id'] for v in found if v['status']=='verified_metadata']
    if matching:
        try:durations=_fetch_durations(matching,api_key,open_fn)
        except Exception:durations={}
        for v in found:
            v['duration_seconds']=durations.get(v['video_id'])
            duration=v['duration_seconds']
            v['video_kind']=('possible_full_game' if duration>=75*60 else 'highlight_or_clip') if duration is not None else 'unknown'
    return found


def _store(db,v):
    db.execute('''INSERT INTO video_discoveries
      (game_id,provider,video_id,title,url,channel,published,checked_utc,status,verification_reason,duration_seconds,video_kind)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
      ON CONFLICT(game_id,provider,video_id) DO UPDATE SET
        title=excluded.title,url=excluded.url,channel=excluded.channel,
        published=excluded.published,checked_utc=excluded.checked_utc,
        status=excluded.status,verification_reason=excluded.verification_reason,
        duration_seconds=excluded.duration_seconds,video_kind=excluded.video_kind''',
      (v['game_id'],v['provider'],v['video_id'],v['title'],v['url'],v['channel'],
       v['published'],datetime.now(timezone.utc).isoformat(),v['status'],
       v['verification_reason'],v.get('duration_seconds'),v.get('video_kind','unknown')))


def reverify_existing(db):
    """Recheck previous discoveries so stale unverified rows cannot enter learning."""
    setup(db)
    rows=db.execute('''SELECT v.game_id,v.provider,v.video_id,v.title,v.published,g.away,g.home,g.game_date
       FROM video_discoveries v LEFT JOIN games g ON g.game_id=v.game_id''').fetchall()
    counts={'verified_metadata':0,'rejected_mismatch':0,'candidate_unverified':0}
    for gid,provider,vid,title,published,away,home,day in rows:
        status,reason=(verify_candidate({'title':title,'published':published},away,home,day)
                       if away and home and day else ('candidate_unverified','game_record_unavailable'))
        db.execute('UPDATE video_discoveries SET status=?,verification_reason=? WHERE game_id=? AND provider=? AND video_id=?',
                   (status,reason,gid,provider,vid))
        counts[status]+=1
    db.commit()
    return counts


def discover(db,api_key,limit=10,opener=None):
    setup(db)
    existing=reverify_existing(db)
    if not api_key:return {'status':'disabled','reason':'Set YOUTUBE_DATA_API_KEY for optional metadata discovery','found':0,'existing_reviewed':existing}
    rows=db.execute("SELECT game_id,game_date,away,home FROM games WHERE state='post' ORDER BY game_date DESC LIMIT ?",(limit,)).fetchall()
    found=0;errors=[];counts={'verified_metadata':0,'rejected_mismatch':0,'candidate_unverified':0}
    for game_id,day,away,home in rows:
        try:
            results=search_youtube(game_id,away,home,day,api_key,opener)
            for v in results:
                _store(db,v)
                counts[v['status']]+=1
            found+=len(results)
        except Exception as exc:errors.append({'game_id':str(game_id),'error':str(exc)[:250]})
    db.commit()
    return {'status':'partial' if errors else 'ok','found':found,'games_checked':len(rows),'classification':counts,
            'existing_reviewed':existing,'errors':errors,
            'note':'Verified means metadata match ONLY. No video downloaded, analyzed, or proven to cover the entire game.'}

def process_authorized_clips(db,max_clips=4,max_frames=600):
    from pbp_learning import index_local_event_clips,setup as setup_pbp
    setup_pbp(db)
    linked=index_local_event_clips(db,CLIPS)
    rows=db.execute('''SELECT game_id,play_id,video_ref,local_path FROM event_videos
       WHERE local_path IS NOT NULL AND (vision_video_id IS NULL OR status!='analyzed')
       ORDER BY game_id,play_id LIMIT ?''',(max_clips,)).fetchall()
    results=[]
    for game_id,play_id,ref,path in rows:
        try:
            p=Path(path).resolve()
            if not p.is_file() or not p.is_relative_to(CLIPS.resolve()):
                raise ValueError('Clip is outside authorized event_clips folder')
            from courtside_vision import process_video
            output=process_video(p,max_frames=max_frames,stride=3)
            db.execute('''UPDATE event_videos SET vision_video_id=?,status='analyzed',
               checked_utc=? WHERE game_id=? AND play_id=? AND video_ref=?''',
               (output['video_id'],datetime.now(timezone.utc).isoformat(),game_id,play_id,ref))
            db.commit()
            results.append({'game_id':game_id,'play_id':play_id,'status':'analyzed'})
        except Exception as exc:
            results.append({'game_id':game_id,'play_id':play_id,'status':'error','error':str(exc)[:250]})
    return {'linked':linked,'processed':results}

def run():
    from nba_archive import connect
    with connect() as db:
        key_file=ROOT/'youtube_api_key'
        key=os.environ.get('YOUTUBE_DATA_API_KEY') or (key_file.read_text().strip() if key_file.is_file() else '')
        discovery=discover(db,key)
        analysis=process_authorized_clips(db)
    return {'discovery':discovery,'analysis':analysis}

if __name__=='__main__':print(json.dumps(run(),indent=2))
