import os
import subprocess

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bash_files', 'launch_seg_queue.sh')


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
