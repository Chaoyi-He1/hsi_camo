import types
import numpy as np
import torch
import torch.nn as nn
import pytest

from models.seg_models import LoRALinear, dice_bce_loss, build_seg_model

TINY_CFG = 'configs/sam2.1/sam2.1_hiera_t.yaml'                              # Hiera-T: 12 blocks, embed 96


def _args(**kw):
    '''main_seg-like namespace for a tiny random-init SAM2.1-T at canvas 128 (no checkpoint download).'''
    a = dict(seg_model='sam2box', sam2_cfg=TINY_CFG, sam2_ckpt='', canvas=128, lora_r=8, lr=1e-4, stem_lr=5e-5,
             weight_decay=1e-4)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _pq(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, n)) * 0.1).astype(np.float32), (rng.standard_normal(3) * 0.1).astype(np.float32)


def test_lora_linear_is_the_base_layer_at_init_and_trains_only_the_adapter():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    torch.testing.assert_close(lora(x), base(x))                              # B = 0 -> unchanged
    assert lora.in_features == 16 and lora.out_features == 24 and lora.scale == 2.0
    lora(x).pow(2).sum().backward()
    assert not base.weight.requires_grad and not base.bias.requires_grad and base.weight.grad is None
    assert lora.lora_B.weight.grad is not None and lora.lora_B.weight.grad.abs().sum() > 0
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.01)
    expected = base(x) + (x @ lora.lora_A.weight.t()) @ lora.lora_B.weight.t() * 2.0
    torch.testing.assert_close(lora(x), expected)


def test_dice_bce_loss_values():
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, :, 2:6, 2:6] = 1.0                                                # image 1 is a false-positive item (empty target)
    good = (mask * 2 - 1) * 20.0                                              # confident and correct logits
    bce, dice = dice_bce_loss(good, mask)
    assert float(bce) < 1e-6 and float(dice) < 1e-3
    bad = -good
    bce_b, dice_b = dice_bce_loss(bad, mask)
    assert float(bce_b) > 10.0 and float(dice_b) > 0.45                       # image 0 dice ~1, image 1 ~0 (smooth) -> ~0.5
    logits = torch.randn(2, 1, 8, 8)
    p = torch.sigmoid(logits)
    ref_dice = (1 - (2 * (p * mask).sum((2, 3)) + 1) / (p.sum((2, 3)) + mask.sum((2, 3)) + 1)).mean()
    ref_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    b, d = dice_bce_loss(logits.to(torch.bfloat16), mask)
    assert b.dtype == torch.float32
    torch.testing.assert_close(d, ref_dice, rtol=2e-2, atol=2e-2)              # bf16 logits, fp32 loss
    torch.testing.assert_close(dice_bce_loss(logits, mask)[1], ref_dice)
    torch.testing.assert_close(dice_bce_loss(logits, mask)[0], ref_bce)


def test_sam2box_rejects_a_canvas_not_divisible_by_32():
    pytest.importorskip('sam2')
    P, q = _pq(3)
    with pytest.raises(AssertionError, match='divisible by 32'):
        build_seg_model(_args(canvas=120), 3, P, q)


@pytest.mark.parametrize('n', [3, 10, 24])
def test_sam2box_builds_and_trains_at_n_channels(n):
    pytest.importorskip('sam2')
    torch.manual_seed(0)
    P, q = _pq(n)
    model = build_seg_model(_args(), n, P, q)
    sam = model.sam
    trunk = sam.image_encoder.trunk
    stem = trunk.patch_embed.proj
    assert stem.in_channels == n + 1 and stem.out_channels == 96
    assert torch.equal(stem.weight[:, n], torch.zeros_like(stem.weight[:, n]))  # box-channel slice starts at zero
    # LoRA exactly on attn.qkv / attn.proj of every block, never on the block-level `proj` of stage-change blocks
    assert model.n_lora == 2 * len(trunk.blocks) == 24
    assert all(isinstance(b.attn.qkv, LoRALinear) and isinstance(b.attn.proj, LoRALinear) for b in trunk.blocks)
    assert not any(isinstance(getattr(b, 'proj', None), LoRALinear) for b in trunk.blocks)
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    assert all(k.startswith('sam.image_encoder.trunk.patch_embed.') or '.lora_' in k or k.startswith('sam.sam_mask_decoder.')
               for k in trainable)
    assert not any('iou_prediction_head' in k or 'pred_obj_score_head' in k for k in trainable)
    assert not any(p.requires_grad for p in sam.sam_prompt_encoder.parameters())
    assert not any(p.requires_grad for p in sam.image_encoder.neck.parameters())

    model.train()
    x = torch.randn(2, n + 1, 128, 128)
    box = torch.tensor([[10.0, 20.0, 70.0, 90.0], [40.0, 30.0, 127.0, 120.0]])
    outs = model(x, box)
    assert isinstance(outs, list) and len(outs) == 1 and outs[0].shape == (2, 1, 128, 128)
    mask = (torch.rand(2, 1, 128, 128) > 0.8).float()
    total, items = model.loss(outs, mask)
    assert set(items) == {'bce', 'dice'} and all(isinstance(v, float) for v in items.values())
    total.backward()
    no_grad = [k for k, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == [], f"trainable parameters without a gradient: {no_grad[:5]}"
    assert stem.weight.grad[:, n].abs().sum() > 0                              # the box channel learns

    groups = model.param_groups(_args())
    assert [g['name'] for g in groups] == ['stem', 'lora', 'decoder']
    assert groups[0]['lr'] == 5e-5 and groups[1]['lr'] == groups[2]['lr'] == 1e-4
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(trainable)
    assert len(groups[0]['params']) == 2 and len(groups[1]['params']) == 2 * model.n_lora
    torch.optim.AdamW(groups)                                                  # extra 'name' key is accepted


def test_sam2box_box_prompt_and_bf16_autocast():
    pytest.importorskip('sam2')
    torch.manual_seed(0)
    P, q = _pq(10)
    model = build_seg_model(_args(), 10, P, q).eval()
    x = torch.randn(1, 11, 128, 128)
    with torch.no_grad():
        a = model(x, torch.tensor([[10.0, 10.0, 60.0, 60.0]]))[0]
        b = model(x, torch.tensor([[60.0, 60.0, 120.0, 120.0]]))[0]
        assert not torch.allclose(a, b)                                        # the prompt reaches the decoder
        torch.testing.assert_close(model(x, torch.tensor([[10.0, 10.0, 60.0, 60.0]]))[0], a)   # eval is deterministic
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outs = model(x, torch.tensor([[10.0, 10.0, 60.0, 60.0]]))
    total, _ = model.loss(outs, torch.zeros(1, 1, 128, 128))
    assert total.dtype == torch.float32 and torch.isfinite(total)
    total.backward()


def test_sam2box_identity_fold_keeps_the_pretrained_rgb_stem():
    '''Zero-shot control: P = I, q = 0 leaves the RGB kernel and bias untouched and adds a zero box slice.'''
    pytest.importorskip('sam2')
    from sam2.build_sam import build_sam2
    torch.manual_seed(0)
    ref = build_sam2(TINY_CFG, None, device='cpu', mode='eval', hydra_overrides_extra=['++model.image_size=128'],
                     apply_postprocessing=False).image_encoder.trunk.patch_embed.proj
    torch.manual_seed(0)
    model = build_seg_model(_args(), 3, np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
    stem = model.sam.image_encoder.trunk.patch_embed.proj
    torch.testing.assert_close(stem.weight[:, :3], ref.weight)
    torch.testing.assert_close(stem.bias, ref.bias)
    assert torch.equal(stem.weight[:, 3], torch.zeros_like(stem.weight[:, 3]))
