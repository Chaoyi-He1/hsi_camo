import argparse
import numpy as np
import pytest
import torch

import main_det
from data_loader.my_dataset import HyperCOD_data
from data_loader.cube_cache import build_cube_cache, default_cache_dir
from models.filter_bank import build_filter_bank
from models.seg_stem import IMAGENET_MEAN, IMAGENET_STD, pseudo_rgb
from models.seg_models import build_front_end, build_seg_model, check_filter_bank

TINY_CFG = 'configs/sam2/sam2_hiera_t.yaml'   # Hiera-T trunk, random init: no checkpoint download in the tests


def _det_args(root, *flags):
    '''Detector args as a checkpoint stores them: main_det's parser defaults on the fixture paths plus the arm flags.'''
    return main_det.get_args_parser().parse_args(['--data-path', str(root), '--cache-dir', default_cache_dir(str(root)), *flags])


RAW = ('--raw-bands',)
EC10 = ('--filter-select', 'uniform', '--num-filters', '12', '--pca-channels', '10')


@pytest.fixture
def frame(synthetic_root):
    '''(root, x [1, 133, 48, 40] fp16 p99-scaled frame as the crop loader hands it over)'''
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0)
    ds = HyperCOD_data(split='train', **main_det.dataset_kwargs(_det_args(root, *RAW)))
    return root, torch.from_numpy(ds[0][0])[None]


def _dataset(root, det):
    return HyperCOD_data(split='train', **main_det.dataset_kwargs(det))


def test_front_end_raw_equals_detector_filter_bank(frame):
    root, x = frame
    det = _det_args(root, *RAW)
    ds = _dataset(root, det)
    fe = build_front_end('raw', det, ds)
    ref, _ = build_filter_bank(det, ds)
    ref.eval()
    y = fe(x)                                                                    # fp16 in, fp32 out
    assert fe.n_out == ds.n_bands == 133 and y.shape == (1, 133, 48, 40) and y.dtype == torch.float32
    assert torch.equal(y, ref(x.float()))
    assert not any(p.requires_grad for p in fe.parameters())
    fe.train()
    assert not fe.training and not fe.fb.training                               # a fixed sensor model stays in eval mode


def test_front_end_ec10_is_fp32_under_bf16_autocast(frame):
    root, x = frame
    det = _det_args(root, *EC10)
    ds = _dataset(root, det)
    fe = build_front_end('ec10', det, ds)
    ref, _ = build_filter_bank(det, ds)
    ref.eval()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = fe(x)
    assert fe.n_out == 10 and y.shape == (1, 10, 48, 40) and y.dtype == torch.float32
    assert torch.equal(y, ref(x.float()))                                       # autocast did not touch the front end


def test_front_end_rgb_is_imagenet_normalised_pseudo_rgb(frame):
    root, x = frame
    det = _det_args(root, *RAW)
    ds = _dataset(root, det)
    fe = build_front_end('rgb', det, ds)
    y = fe(x)
    mean = torch.as_tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.as_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert fe.n_out == 3 and y.dtype == torch.float32
    torch.testing.assert_close(y, (pseudo_rgb(x.float(), ds.wavelens) - mean) / std)


def test_front_end_rejects_a_detector_of_another_arm(frame):
    root, _ = frame
    det_raw, det_ec10 = _det_args(root, *RAW), _det_args(root, *EC10)
    ds_raw, ds_ec = _dataset(root, det_raw), _dataset(root, det_ec10)
    for arm, det, ds in (('ec10', det_raw, ds_raw), ('ec24', det_ec10, ds_ec), ('raw', det_ec10, ds_ec), ('rgb', det_ec10, ds_ec)):
        with pytest.raises(AssertionError):
            build_front_end(arm, det, ds)


def test_front_end_checks_the_detector_checkpoint_buffers(frame):
    root, _ = frame
    det = _det_args(root, *EC10)
    ds = _dataset(root, det)
    fb, _ = build_filter_bank(det, ds)
    state = {f'filter_bank.{k}': v.clone() for k, v in fb.state_dict().items()}
    state['yolo.model.0.conv.weight'] = torch.zeros(1)                          # the rest of a detector state dict is ignored
    assert build_front_end('ec10', det, ds, det_state=state).n_out == 10
    state['filter_bank.mean'] = state['filter_bank.mean'] + 1e-3                # e.g. another detector's whitening
    with pytest.raises(AssertionError, match='mean'):
        build_front_end('ec10', det, ds, det_state=state)
    with pytest.raises(AssertionError):
        check_filter_bank(fb, {k: v for k, v in state.items() if not k.endswith('.std')})


def _seg_args(**kw):
    a = dict(seg_model='sam2unet', sam2unet_cfg=TINY_CFG, sam2unet_ckpt='none', lr=1e-3, stem_lr=1e-4, weight_decay=1e-4)
    a.update(kw)
    return argparse.Namespace(**a)


def _stem_map(n_in, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal((3, n_in))).astype(np.float32), (0.1 * rng.standard_normal(3)).astype(np.float32)


@pytest.mark.parametrize('n_in', [3, 10, 24])
def test_sam2unet_seg_forward_backward(n_in):
    pytest.importorskip('sam2')
    torch.manual_seed(0)
    P, q = _stem_map(n_in)
    model = build_seg_model(_seg_args(), n_in, P, q)
    stem = model.net.encoder.patch_embed.proj
    assert stem.in_channels == n_in + 1 and stem.weight.requires_grad and stem.bias.requires_grad
    assert float(stem.weight.detach()[:, n_in].abs().max()) == 0.0                       # box-channel slice starts at zero
    model.train()
    x = torch.randn(2, n_in + 1, 64, 64)
    box = torch.tensor([[12., 16., 36., 40.]] * 2)                              # canvas px xyxy (unused by SAM2-UNet)
    mask = torch.zeros(2, 1, 64, 64); mask[:, :, 16:40, 12:36] = 1.0
    outs = model(x, box)
    assert len(outs) == 3 and all(o.shape == (2, 1, 64, 64) for o in outs)
    total, items = model.loss(outs, mask)
    assert torch.isfinite(total) and set(items) == {'loss', 'loss_main', 'loss_s16', 'loss_s8'}
    assert all(isinstance(v, float) for v in items.values())
    assert items['loss'] == pytest.approx(items['loss_main'] + items['loss_s16'] + items['loss_s8'], rel=1e-5)
    total.backward()
    assert stem.weight.grad is not None and float(stem.weight.grad[:, n_in].abs().sum()) > 0   # the box slice learns
    named = dict(model.named_parameters())
    trunk = [p for n, p in named.items() if n.startswith('net.encoder.') and '.prompt_learn.' not in n and not n.startswith(model.STEM)]
    adapters = [p for n, p in named.items() if '.prompt_learn.' in n]
    assert trunk and all(not p.requires_grad and p.grad is None for p in trunk)  # Hiera frozen
    assert adapters and all(p.requires_grad and p.grad is not None for p in adapters)
    assert all(p.grad is not None for n, p in named.items() if n.startswith(('net.rfb', 'net.up3', 'net.head', 'net.side')))


def test_sam2unet_seg_stem_is_folded_from_the_trunk_stem():
    pytest.importorskip('sam2')
    from third_party.sam2_unet.sam2unet import build_hiera_trunk
    n_in = 10
    P, q = _stem_map(n_in)
    torch.manual_seed(0)
    rgb_stem = build_hiera_trunk(TINY_CFG, None).patch_embed.proj               # the same random trunk as below
    torch.manual_seed(0)
    model = build_seg_model(_seg_args(), n_in, P, q)
    z = torch.randn(1, n_in, 64, 64)
    rgb = torch.einsum('kc,bchw->bkhw', torch.from_numpy(P), z) + torch.from_numpy(q).view(1, 3, 1, 1)
    with torch.no_grad():
        new = model.net.encoder.patch_embed.proj(torch.cat([z, torch.rand(1, 1, 64, 64)], dim=1))
        old = rgb_stem(rgb)
    # away from the zero-padded border, where the bias fold of q differs (padding is zero in z, not in P z + q)
    torch.testing.assert_close(new[..., 1:-1, 1:-1], old[..., 1:-1, 1:-1], rtol=1e-4, atol=1e-4)


def test_sam2unet_seg_param_groups_and_eval_mode():
    pytest.importorskip('sam2')
    args = _seg_args()
    P, q = _stem_map(10)
    model = build_seg_model(args, 10, P, q)
    groups = model.param_groups(args)
    ids = [id(p) for g in groups for p in g['params']]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == trainable                  # every trainable tensor exactly once
    by = {g['name']: g for g in groups}
    stem = model.net.encoder.patch_embed.proj
    assert {id(p) for p in by['stem']['params']} == {id(stem.weight), id(stem.bias)}
    assert by['stem']['lr'] == 1e-4 and by['stem']['weight_decay'] == 0.0
    assert by['decay']['lr'] == 1e-3 and by['decay']['weight_decay'] == 1e-4 and by['no_decay']['weight_decay'] == 0.0
    assert all(p.ndim > 1 for p in by['decay']['params'])
    assert not any(p.requires_grad for p in model.net.up4.parameters())         # never called upstream
    torch.optim.AdamW(groups, lr=args.lr)
    model.eval()
    with torch.no_grad():
        outs = model(torch.randn(1, 11, 64, 64), torch.zeros(1, 4))
    assert outs[0].shape == (1, 1, 64, 64)
    with pytest.raises(AssertionError):
        model(torch.randn(1, 11, 48, 48), torch.zeros(1, 4))                    # side not a multiple of 32
    with pytest.raises(AssertionError):
        model(torch.randn(1, 10, 64, 64), torch.zeros(1, 4))                    # box channel missing


def test_build_seg_model_checks_name_and_stem_map():
    pytest.importorskip('sam2')
    P, q = _stem_map(10)
    with pytest.raises(AssertionError):
        build_seg_model(_seg_args(seg_model='unet'), 10, P, q)
    with pytest.raises(AssertionError):
        build_seg_model(_seg_args(), 24, P, q)                                   # P fitted for another arm
    with pytest.raises(AssertionError):
        build_seg_model(_seg_args(sam2unet_ckpt='/nonexistent/sam2_hiera_large.pt'), 10, P, q)
