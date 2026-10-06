import fcntl
import os
import shutil
import subprocess
import time

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bash_files', 'launch_seg_queue.sh')
DETS = ('raw133_A', 'sel10g_clean_A', 'sel24g_clean_A')
TOOLS = ('bash', 'readlink', 'dirname', 'mkdir', 'flock', 'setsid', 'nohup', 'date', 'grep', 'tail', 'tee')
# python stand-in: records every call; the epoch query (-c) answers 7; main_seg.py writes its run's final line (or fails
# with STUB_FAIL); main_seg_eval.py does nothing. Builtins only, so it also runs on the PATH without nvidia-smi.
PY_STUB = r'''#!/bin/bash
echo "$*" >> "$STUB_LOG"
if [ "$1" = "-c" ]; then echo 7; exit 0; fi
name=; out=; prev=
for a in "$@"; do
  case "$prev" in --name) name=$a ;; --output_dir) out=$a ;; esac
  prev=$a
done
case "$*" in
  *main_seg.py*)
    [ -n "${STUB_FAIL:-}" ] && { echo "stub crash $name"; exit 3; }
    echo '{"epoch": 0, "final": true}' >> "$out/results_$name.txt"; echo "trained $name" ;;
esac
exit 0
'''
SMI_STUB = '#!/bin/bash\necho -n "${STUB_SMI:-}"\nexit ${STUB_SMI_RC:-0}\n'


def _plan(**env):
    '''The queue's planned run names (LIST_ONLY=1: printed, nothing detached, launched or built).'''
    out = subprocess.run(['bash', SCRIPT], env={**os.environ, 'LIST_ONLY': '1', **env}, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = [l.split() for l in out.stdout.splitlines() if l.strip()]
    assert all(len(l) == 2 and l[1] in ('todo', 'done') for l in lines), out.stdout
    return [l[0] for l in lines]


def test_queue_script_parses():
    assert subprocess.run(['bash', '-n', SCRIPT]).returncode == 0


def test_queue_plan_is_seed_major_with_the_rgb_control():
    names = _plan()
    assert len(names) == 28 and len(set(names)) == 28                       # 3 models x 3 arms x 3 seeds + the control
    assert names[:9] == [f'seg_{m}_{a}_s0' for m in ('sam2unet', 'sam2box', 'zoomnext') for a in ('raw', 'ec10', 'ec24')]
    assert names[-1] == 'seg_sam2unet_rgb_s0'
    assert _plan(MODELS='zoomnext', ARMS='ec10', SEEDS='1', CONTROL_SEEDS='') == ['seg_zoomnext_ec10_s1']
    assert _plan(SEEDS='0', CONTROL_SEEDS='0 1')[-2:] == ['seg_sam2unet_rgb_s0', 'seg_sam2unet_rgb_s1']


@pytest.fixture
def sandbox(tmp_path):
    '''
    A copy of the queue script in an empty repo (it cd's to its own repo root, so nothing touches the real logs/ or
    weights/): the detector prerequisites as empty files, a crop cache index, the python stub as PY and an nvidia-smi
    stub first on PATH (STUB_SMI = its output, STUB_SMI_RC = its exit code). Plan: sam2unet x raw x seeds 0 1, no control.
    Returns (repo, env, run) with run(**env) -> CompletedProcess of the queue in child mode (no detaching).
    '''
    repo = tmp_path / 'repo'
    (repo / 'bash_files').mkdir(parents=True)
    shutil.copy(SCRIPT, repo / 'bash_files' / 'launch_seg_queue.sh')
    (repo / 'results' / 'det').mkdir(parents=True)
    for det in DETS:
        (repo / 'weights' / det).mkdir(parents=True)
        (repo / 'weights' / det / 'model_best').touch()
        for split in ('train', 'val', 'test'):
            (repo / 'results' / 'det' / f'rois_{det}_{split}.json').write_text('{}')
    (tmp_path / 'crops').mkdir()
    (tmp_path / 'crops' / 'index.json').write_text('{}')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, text in (('py', PY_STUB), ('nvidia-smi', SMI_STUB)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('LIST_ONLY', 'SEG_QUEUE_CHILD', 'FORCE_GPU', 'EVAL', 'EVAL_LAST',
                                                            'CACHE_ONLY', 'NUM_WORKERS', 'STUB_SMI', 'STUB_SMI_RC', 'STUB_FAIL')}
    env.update(PY=str(bin_dir / 'py'), PATH=f"{bin_dir}:{os.environ['PATH']}", CROP_CACHE=str(tmp_path / 'crops'),
               DATA_PATH=str(tmp_path / 'data'), CUDA_VISIBLE_DEVICES='0', STUB_LOG=str(tmp_path / 'calls.txt'),
               MODELS='sam2unet', ARMS='raw', SEEDS='0 1', CONTROL_SEEDS='')

    def run(child=True, **extra):
        e = {**env, **({'SEG_QUEUE_CHILD': '1'} if child else {}), **extra}
        return subprocess.run(['bash', str(repo / 'bash_files' / 'launch_seg_queue.sh')], env=e, capture_output=True, text=True,
                              timeout=60)
    return repo, env, run


def _calls(env, what):
    path = env['STUB_LOG']
    return [l for l in open(path).read().splitlines() if what in l] if os.path.exists(path) else []


def _hold(path):
    '''Take an exclusive flock on path (as another queue would); os.close(fd) releases it.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _finish(repo, *names):
    for n in names:
        (repo / 'weights' / n).mkdir(parents=True, exist_ok=True)
        (repo / 'weights' / n / f'results_{n}.txt').write_text('{"epoch": 0, "final": true}\n')


def test_queue_resumes_an_interrupted_run_and_starts_the_others_fresh(sandbox):
    repo, env, run = sandbox
    s0, s1 = 'seg_sam2unet_raw_s0', 'seg_sam2unet_raw_s1'
    (repo / 'weights' / s0).mkdir(parents=True)
    (repo / 'weights' / s0 / 'model_last').touch()                         # s0 was interrupted after a checkpoint
    (repo / 'logs').mkdir()
    (repo / 'logs' / f'{s0}.log').write_text('old line\n')
    out = run(EVAL='0')
    assert out.returncode == 0, out.stdout + out.stderr
    assert f'{s0}: resume from model_last epoch 7' in out.stdout and f'{s1}: start fresh' in out.stdout
    trains = _calls(env, 'main_seg.py')
    assert len(trains) == 2 and f'--resume weights/{s0}/model_last' in trains[0] and '--resume' not in trains[1]
    assert all('--num_workers 8' in t for t in trains)                     # NUM_WORKERS default 8
    log0 = (repo / 'logs' / f'{s0}.log').read_text()
    assert log0.startswith('old line\n') and 'resume from model_last epoch 7' in log0 and f'trained {s0}' in log0   # appended
    assert not _calls(env, 'main_seg_eval.py') and 'EVAL=0: no evaluation' in out.stdout
    # relaunch: both runs finished, nothing is trained again
    out = run(EVAL='0')
    assert out.returncode == 0 and len(_calls(env, 'main_seg.py')) == 2 and out.stdout.count('already finished') == 2


def test_queue_eval_switch(sandbox):
    repo, env, run = sandbox
    _finish(repo, 'seg_sam2unet_raw_s0', 'seg_sam2unet_raw_s1')
    for flags, n_eval in (({}, 2), ({'EVAL_LAST': '0'}, 1), ({'EVAL': ''}, 0), ({'EVAL': '0'}, 0)):
        before = len(_calls(env, 'main_seg_eval.py'))
        out = run(**flags)
        assert out.returncode == 0, out.stdout + out.stderr
        evals = _calls(env, 'main_seg_eval.py')[before:]
        assert len(evals) == n_eval, (flags, evals)
        assert all('--runs seg_sam2unet_raw_s0 seg_sam2unet_raw_s1' in e for e in evals)
    assert '--zero_shot' in _calls(env, 'main_seg_eval.py')[0] and '--ckpt model_last' in _calls(env, 'main_seg_eval.py')[1]


def test_queue_refuses_a_gpu_held_by_another_queue(sandbox):
    repo, env, run = sandbox
    fd = _hold(str(repo / 'logs' / 'seg_queue_gpu0.lock'))
    try:
        for child in (False, True):
            out = run(child=child)
            assert out.returncode == 1 and 'another seg queue holds GPU 0' in out.stdout, out.stdout + out.stderr
        assert not os.path.exists(env['STUB_LOG']) and not (repo / 'logs' / 'seg_queue_gpu0.log').exists()   # nothing ran
        assert run(LIST_ONLY='1', child=False).returncode == 0                # the plan listing needs no lock
        assert run(child=True, CUDA_VISIBLE_DEVICES='1', EVAL='0').returncode == 0   # another GPU is free
    finally:
        os.close(fd)


def test_queue_refuses_a_busy_gpu_unless_forced(sandbox, tmp_path):
    repo, env, run = sandbox
    for child in (False, True):
        out = run(child=child, STUB_SMI='3130234\n')
        assert out.returncode == 1 and 'GPU 0 is busy (compute pid(s) 3130234)' in out.stdout, out.stdout + out.stderr
        assert 'FORCE_GPU=1' in out.stdout
    out = run(STUB_SMI='No devices were found', STUB_SMI_RC='6')           # nvidia-smi failing is no all-clear
    assert out.returncode == 1 and 'nvidia-smi -i 0 failed: No devices were found' in out.stdout
    assert not os.path.exists(env['STUB_LOG'])
    assert run(LIST_ONLY='1', child=False, STUB_SMI='3130234').returncode == 0
    out = run(STUB_SMI='3130234', FORCE_GPU='1', EVAL='0')                 # the expert override
    assert out.returncode == 0 and len(_calls(env, 'main_seg.py')) == 2, out.stdout + out.stderr
    # no nvidia-smi on the PATH at all: the check is skipped
    bare = tmp_path / 'bare'
    bare.mkdir()
    for t in TOOLS:
        os.symlink(shutil.which(t), bare / t)
    _finish(repo, 'seg_sam2unet_raw_s0', 'seg_sam2unet_raw_s1')
    out = run(PATH=str(bare), EVAL='0')
    assert out.returncode == 0 and 'queue done' in out.stdout, out.stdout + out.stderr


def test_queue_leaves_a_run_another_queue_is_training(sandbox):
    repo, env, run = sandbox
    fd = _hold(str(repo / 'weights' / 'seg_sam2unet_raw_s0' / '.queue.lock'))
    try:
        out = run(EVAL='0')
    finally:
        os.close(fd)
    assert out.returncode == 0 and 'seg_sam2unet_raw_s0 is being trained by another seg queue, skipping' in out.stdout
    trains = _calls(env, 'main_seg.py')
    assert len(trains) == 1 and '--name seg_sam2unet_raw_s1' in trains[0]


def test_evaluating_queue_waits_for_a_training_queue(sandbox, tmp_path):
    repo, env, run = sandbox
    _finish(repo, 'seg_sam2unet_raw_s0', 'seg_sam2unet_raw_s1')
    fd = _hold(str(repo / 'logs' / 'seg_train_gpu1.lock'))                 # a GPU-1 queue still training
    log = tmp_path / 'queue.out'
    with open(log, 'w') as f:
        proc = subprocess.Popen(['bash', str(repo / 'bash_files' / 'launch_seg_queue.sh')], env={**env, 'SEG_QUEUE_CHILD': '1'},
                                stdout=f, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while 'waiting for the seg queue' not in log.read_text() and time.time() - t0 < 30:
            time.sleep(0.1)
        assert 'logs/seg_train_gpu1.lock' in log.read_text() and proc.poll() is None
        assert not _calls(env, 'main_seg_eval.py')                          # no evaluation while GPU 1 trains
    finally:
        os.close(fd)
    assert proc.wait(timeout=60) == 0
    assert len(_calls(env, 'main_seg_eval.py')) == 2 and 'queue done' in log.read_text()
