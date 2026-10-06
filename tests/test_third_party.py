import hashlib
import os
import re
import subprocess
import sys
import torch
import torch.nn.functional as F

from third_party.sam2_unet.sam2unet import structure_loss

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, 'third_party', 'sam2_unet')
SAM2UNET_COMMIT = '01598e5e9912ffb23f965ecbebf4d1dfecbaa56e'
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'   # upstream LICENSE at that commit


def test_vendored_sam2unet_carries_licence_and_notice():
    with open(os.path.join(VENDOR, 'LICENSE'), 'rb') as f:
        lic = f.read()
    assert hashlib.sha256(lic).hexdigest() == LICENSE_SHA256 and b'Apache License' in lic[:200]
    with open(os.path.join(VENDOR, 'NOTICE')) as f:
        notice = f.read()
    assert SAM2UNET_COMMIT in notice and 'Apache License 2.0' in notice and 'CHANGED (hsi_camo)' in notice
    # the unmarked edits are listed too: the dropped `del model.<submodule>` lines, whitespace, the up4 comment
    assert 'del model.' in notice and 'trailing whitespace' in notice and 'up4' in notice
    files = sorted(os.listdir(VENDOR))
    # SAM2-UNet's own `sam2` package copy must never be vendored (it shadows the pip sam2), nor any weights
    assert 'sam2' not in files and 'sam2_configs' not in files
    assert not [f for f in files if f.endswith(('.pt', '.pth', '.pyd'))]
    with open(os.path.join(REPO, '.gitignore')) as f:
        assert 'third_party/zoomnext/' in f.read().splitlines()


def _wbce_wiou(pred, mask):
    '''structure_loss written out per pixel: weighted BCE + weighted IoU, mean over the batch.'''
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    bce = -(mask * F.logsigmoid(pred) + (1 - mask) * F.logsigmoid(-pred))                  # [B, 1, h, w]
    wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    p = torch.sigmoid(pred)
    inter, union = (p * mask * weit).sum(dim=(2, 3)), ((p + mask) * weit).sum(dim=(2, 3))
    return wbce, 1 - (inter + 1) / (union - inter + 1)


def test_structure_loss_is_weighted_bce_plus_weighted_iou():
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 64, 64) * 3
    mask = torch.zeros(2, 1, 64, 64); mask[:, :, 20:44, 10:30] = 1.0
    wbce, wiou = _wbce_wiou(pred, mask)
    torch.testing.assert_close(structure_loss(pred, mask), (wbce + wiou).mean(), rtol=1e-5, atol=1e-6)
    # upstream's reduce='none' means reduction='mean': the plain mean BCE, which differs from the weighted one
    legacy = F.binary_cross_entropy_with_logits(pred, mask) + wiou.mean()
    torch.testing.assert_close(structure_loss(pred, mask, legacy_bce=True), legacy, rtol=1e-5, atol=1e-6)
    assert abs(float(structure_loss(pred, mask)) - float(legacy)) > 1e-3
    good = (mask * 2 - 1) * 20                                                              # confident and right
    assert float(structure_loss(good, mask)) < 0.05 < float(structure_loss(-good, mask))


def test_setup_versions_guard_covers_torchvision():
    '''setup_third_party.sh's versions() guard (compared before / after the installs) reports numpy, cv2, torch and torchvision.'''
    import cv2
    import numpy
    import torchvision
    with open(os.path.join(REPO, 'bash_files', 'setup_third_party.sh')) as f:
        line = next(l for l in f.read().splitlines() if re.match(r'versions\(\)\s*\{', l))
    out = subprocess.run(['bash', '-c', f'{line}\nversions'], env={**os.environ, 'PY': sys.executable, 'CUDA_VISIBLE_DEVICES': ''},
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [numpy.__version__, cv2.__version__, torch.__version__, torchvision.__version__]
