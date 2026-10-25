import plistlib
from pathlib import Path
from video_autostart import agent_definition,install,LABEL

def test_agent_stable_paths(tmp_path):
    source=tmp_path/'Application Support'/'Courtside IQ'
    config=agent_definition(source/'.venv/bin/python',source,tmp_path)
    assert config['RunAtLoad'] and config['KeepAlive']
    assert str(tmp_path/'.courtsideiq') in config['StandardOutPath']
    assert config['ProgramArguments'][-1].endswith('video_autopilot.py')

def test_install_idempotent_and_retires_old_video_only(tmp_path):
    calls=[]
    class Result: returncode=0;stderr=b''
    def runner(args,**kwargs):calls.append(args);return Result()
    agents=tmp_path/'Library'/'LaunchAgents';agents.mkdir(parents=True)
    old=agents/'com.courtsideiq.automaticvideo.plist';old.write_text('old')
    archive=agents/'com.courtsideiq.nbaarchive.plist';archive.write_text('keep')
    py=tmp_path/'python';py.write_text('')
    src=tmp_path/'source';src.mkdir()
    result=install(python=py,source=src,home=tmp_path,runner=runner)
    assert result['status']=='installed'
    assert not old.exists() and archive.read_text()=='keep'
    plist=agents/(LABEL+'.plist')
    assert plistlib.loads(plist.read_bytes())['KeepAlive']
    assert install(python=py,source=src,home=tmp_path,runner=runner)['status']=='already_installed'
    assert len(calls)==3
