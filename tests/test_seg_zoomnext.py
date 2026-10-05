import os
import math
import types
import numpy as np
import torch
import pytest

from models.seg_models import ZOOMNEXT_DIR, ZoomNeXtSeg, build_seg_model, import_zoomnext, no_cuda_query, zoomnext_ual_coef
from models.seg_stem import fold_stem

# ZoomNeXt is fetched by bash_files/setup_third_party.sh into git-ignored third_party/zoomnext and needs einops
requires_zoomnext = pytest.mark.skipif(not os.path.isfile(os.path.join(ZOOMNEXT_DIR, 'methods', 'zoomnext', 'zoomnext.py')),
                                       reason="third_party/zoomnext missing: run bash_files/setup_third_party.sh")


def _args(**kw):
    '''main_seg-like namespace for a random-init ZoomNeXt-B2 at canvas 128 (no checkpoint download).'''
    a = dict(seg_model='zoomnext', zoomnext_ckpt='', canvas=128, lr=1e-4, stem_lr=5e-5, weight_decay=1e-4)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _pq(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, n)) * 0.1).astype(np.float32), (rng.standard_normal(3) * 0.1).astype(np.float32)


def test_ual_coef_ramps_from_zero_to_one():
    assert zoomnext_ual_coef(0.0) == 0.0 and zoomnext_ual_coef(1.0) == 1.0
    assert zoomnext_ual_coef(0.5) == pytest.approx(0.5)
    assert zoomnext_ual_coef(-1.0) == 0.0 and zoomnext_ual_coef(2.0) == 1.0
    assert zoomnext_ual_coef(0.25) == pytest.approx((1 - math.cos(math.pi * 0.25)) / 2)


@requires_zoomnext
def test_zoomnext_rejects_a_canvas_not_divisible_by_64():
    pytest.importorskip('einops')
    P, q = _pq(3)
    with pytest.raises(AssertionError, match='divisible by 64'):
        build_seg_model(_args(canvas=96), 3, P, q)


@requires_zoomnext
@pytest.mark.parametrize('n', [3, 10, 24])
def test_zoomnext_builds_and_trains_at_n_channels(n):
    pytest.importorskip('einops')
    torch.manual_seed(0)
    P, q = _pq(n)
    model = build_seg_model(_args(), n, P, q)
    assert isinstance(model, ZoomNeXtSeg)
    stem = model.net.encoder.patch_embed1.proj
    assert stem.in_channels == n + 1 and stem.out_channels == 64 and stem.stride == (4, 4)
    assert torch.equal(stem.weight[:, n], torch.zeros_like(stem.weight[:, n]))  # box-channel slice starts at zero
    assert all(p.requires_grad for p in model.parameters())                      # full fine-tune, stem unfrozen

    model.train()
    x = torch.randn(2, n + 1, 128, 128)
    outs = model(x, torch.zeros(2, 4))
    assert isinstance(outs, list) and len(outs) == 1 and outs[0].shape == (2, 1, 128, 128)
    mask = (torch.rand(2, 1, 128, 128) > 0.8).float()
    model.set_progress(0.3)
    total, items = model.loss(outs, mask)
    assert set(items) == {'bce', 'ual', 'ual_coef'} and all(isinstance(v, float) for v in items.values())
    assert items['ual_coef'] == pytest.approx(zoomnext_ual_coef(0.3))
    assert float(total.detach()) == pytest.approx(items['bce'] + items['ual_coef'] * items['ual'], rel=1e-5)
    total.backward()
    assert stem.weight.grad is not None and stem.weight.grad[:, n].abs().sum() > 0   # the box channel learns

    groups = model.param_groups(_args(encoder_lr_mult=0.1))
    assert [g['name'] for g in groups] == ['stem', 'encoder', 'decoder']
    assert len(groups[0]['params']) == 4                                         # patch_embed1.proj.{weight,bias}, .norm.{weight,bias}
    assert groups[0]['lr'] == 5e-5 and groups[1]['lr'] == pytest.approx(1e-5) and groups[2]['lr'] == 1e-4
    assert model.param_groups(_args())[1]['lr'] == 1e-4                           # default: one lr for the whole network (spec §7)
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    torch.optim.AdamW(groups)


@requires_zoomnext
def test_zoomnext_progress_zero_is_plain_bce_and_eval_is_deterministic():
    pytest.importorskip('einops')
    torch.manual_seed(0)
    P, q = _pq(10)
    model = build_seg_model(_args(), 10, P, q)
    assert model.progress == 1.0                                                  # ZoomNeXt's own default: the full UAL weight
    model.set_progress(0.0)
    logits = [torch.randn(1, 1, 128, 128)]
    mask = torch.zeros(1, 1, 128, 128)
    total, items = model.loss(logits, mask)
    assert items['ual_coef'] == 0.0
    torch.testing.assert_close(total, torch.nn.functional.binary_cross_entropy_with_logits(logits[0], mask))
    model.eval()
    x = torch.randn(1, 11, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], model(x, torch.zeros(1, 4))[0])
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outs = model(torch.randn(2, 11, 128, 128))
    total, _ = model.loss(outs, torch.zeros(2, 1, 128, 128))
    assert total.dtype == torch.float32 and torch.isfinite(total)
    with pytest.raises(AssertionError, match='batch of at least 2'):
        model(torch.randn(1, 11, 128, 128))                                      # SimpleASPP's pooled BatchNorm branch


@requires_zoomnext
def test_zoomnext_loads_the_cod_checkpoint_before_folding_the_stem(tmp_path):
    '''A checkpoint shaped like pvtv2-b2-zoomnext.pth (normalizer.* present, num_batches_tracked absent) loads, and the
    stem is the fold of the CHECKPOINT's RGB kernel.'''
    pytest.importorskip('einops')
    Net = import_zoomnext()
    torch.manual_seed(1)
    with no_cuda_query():
        ref = Net(pretrained=False, num_frames=1, input_norm=True)
    sd = {k: v for k, v in ref.state_dict().items() if not k.endswith('num_batches_tracked')}
    assert any(k.startswith('normalizer.') for k in sd)
    path = str(tmp_path / 'pvtv2-b2-zoomnext.pth')
    torch.save(sd, path)

    torch.manual_seed(2)                                                          # a different random init underneath
    P, q = _pq(10)
    model = build_seg_model(_args(zoomnext_ckpt=path), 10, P, q)
    own = model.net.state_dict()
    for k in ('encoder.block1.0.attn.q.weight', 'encoder.patch_embed2.proj.weight', 'encoder.patch_embed1.norm.weight',
              'tra_5.conv1x1_1.conv.weight', 'tra_5.conv1x1_1.bn.running_var', 'predictor.2.weight'):
        torch.testing.assert_close(own[k], sd[k])
    folded = fold_stem(ref.encoder.patch_embed1.proj, P, q, n_extra=1)
    torch.testing.assert_close(own['encoder.patch_embed1.proj.weight'], folded.weight.data)
    torch.testing.assert_close(own['encoder.patch_embed1.proj.bias'], folded.bias.data)
    assert not any(k.startswith('normalizer.') for k in own)

    bad = dict(sd)
    bad['encoder.block1.0.attn.q.weight'] = torch.zeros(3, 3)
    torch.save(bad, path)
    with pytest.raises(AssertionError, match='shape mismatch'):
        build_seg_model(_args(zoomnext_ckpt=path), 10, P, q)
