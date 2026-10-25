"""Idempotently install the persistent video LaunchAgent on app startup.

Keeps all paths outside the replaceable .app bundle. Never changes the already
installed com.courtsideiq.nbaarchive collector.
"""
from __future__ import annotations
import os, plistlib, subprocess, sys
from pathlib import Path

LABEL='com.courtsideiq.videoautopilot'

def agent_definition(python,source,home):
    root=Path(home)/'.courtsideiq'
    return {'Label':LABEL,
        'ProgramArguments':[str(python),'-u',str(Path(source)/'video_autopilot.py')],
        'WorkingDirectory':str(source),'RunAtLoad':True,'KeepAlive':True,
        'StandardOutPath':str(root/'video_autopilot.log'),
        'StandardErrorPath':str(root/'video_autopilot_errors.log'),
        'ProcessType':'Background'}

def install(python=None,source=None,home=None,runner=None):
    python=Path(python or sys.executable).resolve()
    source=Path(source or Path(__file__).parent).resolve()
    home=Path(home or Path.home()).expanduser()
    runner=runner or subprocess.run
    root=home/'.courtsideiq';root.mkdir(parents=True,exist_ok=True)
    agents=home/'Library'/'LaunchAgents';agents.mkdir(parents=True,exist_ok=True)
    plist=agents/(LABEL+'.plist')
    desired=plistlib.dumps(agent_definition(python,source,home))
    # Retire the previous six-hour video service to prevent duplicate video workers.
    legacy=agents/'com.courtsideiq.automaticvideo.plist'
    if legacy.is_file():
        runner(['/bin/launchctl','bootout',f'gui/{os.getuid()}',str(legacy)],capture_output=True,check=False)
        legacy.unlink(missing_ok=True)
    # No restart on every app launch, only when settings or interpreter change.
    if plist.is_file() and plist.read_bytes()==desired:
        return {'status':'already_installed','plist':str(plist)}
    tmp=plist.with_suffix('.plist.tmp');tmp.write_bytes(desired);os.replace(tmp,plist)
    uid=os.getuid()
    runner(['/bin/launchctl','bootout',f'gui/{uid}',str(plist)],capture_output=True,check=False)
    proc=runner(['/bin/launchctl','bootstrap',f'gui/{uid}',str(plist)],capture_output=True,check=False)
    if proc.returncode:
        return {'status':'error','detail':proc.stderr.decode(errors='replace')[:400] if isinstance(proc.stderr,bytes) else str(proc.stderr)[:400]}
    return {'status':'installed','plist':str(plist)}
