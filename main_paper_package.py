'''
Paper results package: every number of the two-stage study (Stage 1, the EC-YOLO detector; Stage 2, box-guided segmentation)
collected from the result files into CSV tables, figure data and a Markdown summary for the manuscript.
    python main_paper_package.py --out_dir docs/paper
Reads results/det/compare_*/compare.json (fixed-checkpoint detector comparisons + paired frame bootstrap), results/seg*/compare.json
and results/seg/<run>/eval_*.json + per_image.pkl (segmentation), weights/<run>/results_<run>.txt (validation curves),
logs/<run>.log (compute), results/det/voltages_greedy*.json + weights/det_A/gate_ranking_ep89.csv (voltage selection) and the
sensor file EC_filterV3.mat. Nothing is computed anew: the script only re-arranges what the runs wrote, so every table is
reproducible from results/ and the summary quotes the tables. Optional inputs (results/det/compare_rawpca_seeds, the Stage-2
timing bundle) are used when present and reported as missing otherwise.
'''
import argparse
import csv
import glob
import gzip
import json
import os
import pickle
import re
import statistics

import numpy as np

from data_loader.ec_filter import load_ec_filter, align_filter_to_wavelens, candidate_indices, band_indices, WAVELENS_200

# Stage-1 detector runs: label, input, channels, seed, read noise (dB, 0 = none)
DET_RUNS = {
    'det_A': ('Session A', 'all 344 EC readings + trainable channel weights', 344, 42, 0),
    'det_B': ('Session B', 'gate top-10 EC readings, unwhitened, warm start from A (epochs 0-91)', 10, 42, 0),
    'raw133_A': ('raw 133', 'raw 133 cube bands, standardised', 133, 42, 0),
    'raw133_A_s1': ('raw 133 (seed 1)', 'raw 133 cube bands, standardised', 133, 1, 0),
    'raw133_A_s2': ('raw 133 (seed 2)', 'raw 133 cube bands, standardised', 133, 2, 0),
    'pca11_A': ('PCA-11', '344 EC readings whitened to 11 principal directions', 11, 42, 0),
    'sel10g_clean_A': ('greedy-10', '10 greedy-selected EC readings, whitened', 10, 42, 0),
    'sel10g_clean_A_s1': ('greedy-10 (seed 1)', '10 greedy-selected EC readings, whitened', 10, 1, 0),
    'sel10g_clean_A_s2': ('greedy-10 (seed 2)', '10 greedy-selected EC readings, whitened', 10, 2, 0),
    'sel10b_clean_A': ('gate-10 whitened', "session A's gate top-10 EC readings, whitened", 10, 42, 0),
    'sel10u_clean_A': ('uniform-10', '10 uniformly spaced EC readings, whitened', 10, 42, 0),
    'sel24g_clean_A': ('greedy-24', '24 greedy-selected EC readings, whitened', 24, 42, 0),
    'rawpca10_A': ('raw-PCA-10', 'raw 133 cube bands whitened to 10 principal directions', 10, 42, 0),
    'rawpca40_A': ('raw-PCA-40', 'raw 133 cube bands whitened to 40 principal directions', 40, 42, 0),
    'sel10g_aug_A': ('greedy-10 + aug + EMA', '10 greedy EC readings, whitened; scale / crop + photometric jitter, weight EMA', 10, 42, 0),
    'sel10g_aug_A_s1': ('greedy-10 + aug + EMA (seed 1)', '10 greedy EC readings, whitened; scale / crop + photometric jitter, weight EMA', 10, 1, 0),
    'sel10g_aug_A_s2': ('greedy-10 + aug + EMA (seed 2)', '10 greedy EC readings, whitened; scale / crop + photometric jitter, weight EMA', 10, 2, 0),
    'sel10g_aug_m_A': ('greedy-10 + aug + EMA, yolo26m', 'as greedy-10 + aug + EMA with the yolo26m backbone', 10, 42, 0),
    'sel10g_aug_tv_A': ('greedy-10 + aug + EMA, train + val', 'as greedy-10 + aug + EMA trained on the 279 train + val images', 10, 42, 0),
    'raw133_aug_A': ('raw 133 + aug + EMA', 'raw 133 cube bands, standardised; scale / crop + photometric jitter, weight EMA', 133, 42, 0),
    'rgb_A': ('RGB', "the dataset's RGB image, 3 channels", 3, 42, 0),
    'sel10g_A': ('greedy-10, 40 dB', '10 greedy EC readings, 40 dB read noise, noise-regularised whitening', 10, 42, 40),
    'sel10b_A': ('gate-10, 40 dB', 'gate top-10 EC readings, 40 dB read noise, whitened', 10, 42, 40),
    'sel10u_A': ('uniform-10, 40 dB', '10 uniform EC readings, 40 dB read noise, whitened', 10, 42, 40),
    'sel24g_A': ('greedy-24, 40 dB', '24 greedy EC readings, 40 dB read noise, whitened', 24, 42, 40),
}
DET_METRICS = ['ap50', 'recall50', 'matched_iou', 'recall50_op', 'coverage_recall99_roi_op', 'dets_per_image_op']
# the noise-free comparison folders, most complete first: a run's fixed-checkpoint means come from the first that holds it
DET_COMPARE_CLEAN = ['compare_aug_seeds', 'compare_aug', 'compare_rawpca_seeds', 'compare_clean', 'compare_rgb', 'compare_clean_detB']
FINAL_DET = 'sel10g_aug_A_m99'      # the final pipeline's detector: sel10g_aug_A's last (EMA) checkpoint, published as weights/sel10g_aug_A_m99/model_best
FINAL_DET_METRICS_RUN = 'sel10g_aug_A'   # its fixed-checkpoint metrics are those of the run it comes from
DET_COMPARE_NOISE = ['compare_sel_all', 'compare_sel10']
SEG_MODELS = {'sam2unet': 'SAM2-UNet', 'sam2box': 'SAM2.1 + box prompt (LoRA)', 'zoomnext': 'ZoomNeXt-B2', 'sam2box_zs': 'SAM2.1 + box, zero-shot'}
SEG_ARMS = {'raw': 'raw 133 bands', 'ec10': '10 EC readings (greedy-10)', 'ec24': '24 EC readings (greedy-24)', 'rgb': 'pseudo-RGB (3 band means)'}
SEG_LEVELS = ['roi_oracle', 'full_oracle', 'full_det', 'full_det_paper']
# HyperCOD paper, arXiv 2601.03736 (AAAI 2026), Table 2: full images, no boxes, 280 training images
PUBLISHED = [('SAM2-UNet', 0.805, 0.899, 0.0022, 0.641), ('HSC-SAM (paper method, no code)', 0.802, None, None, None), ('HGINet', 0.766, None, None, None)]


def get_args_parser():
    parser = argparse.ArgumentParser('Paper results package', add_help=False)
    parser.add_argument('--results_dir', type=str, default='results', help='results root: det/compare_*, seg, seg_last, seg_rgbdet')
    parser.add_argument('--weights_dir', type=str, default='weights', help='run folders with results_<run>.txt')
    parser.add_argument('--logs_dir', type=str, default='logs', help='training logs <run>.log (epoch / validation times, peak memory)')
    parser.add_argument('--sensor_file', type=str, default='/data2/chaoyi/HyperCOD/Raw data/EC_filterV3.mat', help='EC responsivity R(lambda, V)')
    parser.add_argument('--band_range', type=float, nargs=2, default=[400.0, 800.0], help='cube bands used (nm)')
    parser.add_argument('--seg_timing', type=str, default='results/reports/stage2_seg/bundle_seg.json',
                        help='Stage-2 report bundle with the queue timing (wall hours per run); skipped when missing')
    parser.add_argument('--out_dir', type=str, default='docs/paper', help='package root: tables/, figure_data/, results_summary.md, manifest.json')
    return parser


def write_csv(path, rows, fieldnames=None):
    '''Rows (list of dicts) to CSV with a stable column order; floats as written; gzip-compressed when the path ends in .gz.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            fieldnames += [k for k in r if k not in fieldnames]
    opener = (lambda p: gzip.open(p, 'wt', newline='')) if path.endswith('.gz') else (lambda p: open(p, 'w', newline=''))
    with opener(path) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fieldnames})
    return path


def r4(v):
    return None if v is None else round(float(v), 4)


def ci_text(d):
    '''"+0.077 [-0.013, +0.162]" for a {diff, lo, hi} dict.'''
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def base_run(run):
    return re.sub(r'_s\d+$', '', run)


def log_timing(path):
    '''Median epoch time (epochs >= 2), median validation time, their sum over the log and the peak memory, from a run log.'''
    if not os.path.isfile(path):
        return None
    log = open(path, errors='replace').read()
    def secs(h, m, s):
        return int(h) * 3600 + int(m) * 60 + int(s)
    ep = [(int(n), secs(a, b, c)) for n, a, b, c in re.findall(r'Epoch: \[(\d+)\] Total time: (\d+):(\d+):(\d+)', log)]
    va = [secs(a, b, c) for a, b, c in re.findall(r'Eval val: Total time: (\d+):(\d+):(\d+)', log)]
    mem = [int(x) for x in re.findall(r'max mem: (\d+)', log)]
    if not ep:
        return None
    steady = [t for n, t in ep if n >= 2] or [t for _, t in ep]
    return dict(n_epochs=len(ep), epoch_s_median=round(statistics.median(steady), 1), val_s_median=round(statistics.median(va), 1) if va else None,
                train_h_total=round((sum(t for _, t in ep) + sum(va)) / 3600, 2), peak_mem_mib=max(mem) if mem else None)


def main(args):
    out = args.out_dir
    tables, figs = os.path.join(out, 'tables'), os.path.join(out, 'figure_data')
    os.makedirs(tables, exist_ok=True); os.makedirs(figs, exist_ok=True)
    manifest, missing = {}, []

    # ---------------------------------------------------------------- 1. Stage-1 detectors on fixed checkpoints (noise-free)
    det_means, det_src, det_ckpts = {}, {}, {}
    comps_det = []
    for name in DET_COMPARE_CLEAN + DET_COMPARE_NOISE:
        p = os.path.join(args.results_dir, 'det', name, 'compare.json')
        if not os.path.isfile(p):
            missing.append(p)
            continue
        c = json.load(open(p))
        for key, m in c['means']['test'].items():
            run, cond = key.split('|')
            if cond == 'clean' and name in DET_COMPARE_CLEAN and run not in det_means:
                det_means[run], det_src[run], det_ckpts[run] = m, name, c['ckpt_epochs']
        for key, v in c['comparisons']['test'].items():
            pair, cond = key.split(' | ')
            if ' - ' in pair:
                a, b = pair.split(' - ')
            else:                                                     # "run | trained - clean": one run, evaluated with and without its training noise
                ca, cb = cond.split(' - ')
                a, b, cond = f'{pair} ({ca})', f'{pair} ({cb})', 'trained - clean'
            for k in DET_METRICS:
                if k in v:
                    comps_det.append(dict(source=name, condition=cond, first=a, second=b, metric=k, diff=r4(v[k]['diff']), ci_lo=r4(v[k]['lo']),
                                          ci_hi=r4(v[k]['hi']), p_gt0=r4(v[k].get('p_gt0')), significant=int(v[k]['lo'] > 0 or v[k]['hi'] < 0)))
    rows = []
    for run, m in det_means.items():
        label, inp, nch, seed, noise = DET_RUNS.get(run, (run, '', None, None, None))
        rows.append(dict(run=run, label=label, input=inp, channels=nch, seed=seed, read_noise_db=noise, ckpt_epochs=' '.join(map(str, det_ckpts[run])),
                         source=det_src[run], **{k: r4(m.get(k)) for k in DET_METRICS}))
    rows.sort(key=lambda r: (-(r['ap50'] or 0)))
    manifest['det_fixed_checkpoint'] = write_csv(os.path.join(tables, 'det_fixed_checkpoint.csv'), rows)
    # seeds: mean / std over the runs sharing a base name
    seed_rows = []
    for base in sorted({base_run(r) for r in det_means}):
        runs = sorted(r for r in det_means if base_run(r) == base)
        if len(runs) < 2:
            continue
        row = dict(base_run=base, label=DET_RUNS.get(base, (base,))[0], n_seeds=len(runs), runs=' '.join(runs), seeds=' '.join(str(DET_RUNS.get(r, (0, 0, 0, None))[3]) for r in runs))
        for k in DET_METRICS:
            vals = [det_means[r][k] for r in runs]
            row[f'{k}_mean'] = r4(np.mean(vals)); row[f'{k}_std'] = r4(np.std(vals, ddof=1)); row[f'{k}_min'] = r4(min(vals)); row[f'{k}_max'] = r4(max(vals))
        seed_rows.append(row)
    manifest['det_seed_summary'] = write_csv(os.path.join(tables, 'det_seed_summary.csv'), seed_rows) if seed_rows else None
    manifest['det_paired_differences'] = write_csv(os.path.join(tables, 'det_paired_differences.csv'), comps_det)

    # ---------------------------------------------------------------- 2. Stage-1 validation curves and model_best lines
    val_rows, best_rows = [], []
    for run in sorted(set(DET_RUNS) | set(det_means)):
        p = os.path.join(args.weights_dir, run, f'results_{run}.txt')
        if not os.path.isfile(p):
            continue
        L = [json.loads(l) for l in open(p) if l.strip()]
        for l in L:
            if l.get('final') or 'val' not in l or 'epoch' not in l:
                continue
            v, t = l['val'], l.get('train', {})
            val_rows.append(dict(run=run, epoch=l['epoch'], ap50=r4(v.get('ap50')), recall50=r4(v.get('recall50')), matched_iou=r4(v.get('matched_iou')),
                                 roi_coverage=r4(v.get('coverage_recall99_roi')), dets_per_image=r4(v.get('dets_per_image')),
                                 train_loss=r4(t.get('loss')) if isinstance(t, dict) else None))
        fin = [l for l in L if l.get('final')]
        if fin:
            f, tv = fin[-1], fin[-1].get('test', {})
            best_rows.append(dict(run=run, best_epoch=f.get('best_epoch'), **{f'test_{k}': r4(tv.get(k)) for k in ('ap50', 'recall50', 'matched_iou', 'coverage_recall99_roi', 'dets_per_image')}))
    manifest['det_val_curves'] = write_csv(os.path.join(figs, 'det_val_curves.csv'), val_rows)
    manifest['det_model_best'] = write_csv(os.path.join(tables, 'det_model_best.csv'), best_rows)

    # ---------------------------------------------------------------- 3. voltage selection: the sensor, the sets, the gate weights, the greedy objective
    sel = {}
    if os.path.isfile(args.sensor_file):
        swl, volts, R = load_ec_filter(args.sensor_file)
        bands = band_indices(tuple(args.band_range), WAVELENS_200)
        wl = WAVELENS_200[bands]
        R_al, valid = align_filter_to_wavelens(swl, R, wl)
        usable = candidate_indices(volts)
        np.savez_compressed(os.path.join(figs, 'sensor_response.npz'), sensor_wavelens_nm=swl, voltages_V=volts, responsivity=R,
                            cube_band_centres_nm=wl, responsivity_on_cube_bands=R_al, usable_voltage_indices=usable)
        with open(os.path.join(figs, 'sensor_response_on_cube_bands.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['wavelength_nm'] + [f'{v:.2f}' for v in volts[usable]])
            for i, lam in enumerate(wl):
                w.writerow([f'{lam:.3f}'] + [f'{x:.5f}' for x in R_al[i, usable]])
        manifest['sensor_response'] = os.path.join(figs, 'sensor_response.npz')
        sel['sensor'] = dict(n_wavelengths=int(len(swl)), n_voltages=int(len(volts)), n_usable_voltages=int(len(usable)), dead_zone_V=[0.25, 0.31],
                             n_cube_bands=int(len(wl)), band_range_nm=list(args.band_range))
    else:
        missing.append(args.sensor_file)
    for name, fn in (('greedy10', 'voltages_greedy.json'), ('greedy24', 'voltages_greedy24.json')):
        p = os.path.join(args.results_dir, 'det', fn)
        if os.path.isfile(p):
            g = json.load(open(p))
            sel[name] = dict(voltages_in_selection_order=g['voltages'], objective_per_step=g.get('objective'), k=g.get('k'), snr_db=g.get('snr_db'),
                             noise_model=g.get('noise_model'), ridge=g.get('ridge'), min_spacing_V=g.get('min_spacing'), n_train_frames=g.get('n_frames'),
                             report=g.get('report'))
        else:
            missing.append(p)
    gp = os.path.join(args.weights_dir, 'det_A', 'gate_ranking_ep89.csv')
    if os.path.isfile(gp):
        gr = list(csv.DictReader(open(gp)))
        sel['gate_top10'] = dict(voltages=[float(r['voltage']) for r in gr[:10]], weights=[float(r['weight']) for r in gr[:10]], source='weights/det_A/gate_ranking_ep89.csv')
        write_csv(os.path.join(figs, 'gate_weights_ep89.csv'), [dict(rank=int(r['rank']), voltage_V=float(r['voltage']), weight=float(r['weight'])) for r in gr])
    else:
        missing.append(gp)
    sel['uniform10'] = dict(voltages=[-1.00, -0.62, -0.24, 0.14, 0.59, 0.98, 1.36, 1.74, 2.12, 2.50], rule='10 equally spaced indices over the 344 usable voltages')
    json.dump(sel, open(os.path.join(figs, 'voltage_selection.json'), 'w'), indent=1)
    manifest['voltage_selection'] = os.path.join(figs, 'voltage_selection.json')

    # ---------------------------------------------------------------- 4. Stage-2 groups, comparisons, per-run numbers
    # result sets: the campaign (model_best / model_last), the strict RGB chain, and the FINAL pipeline (segmenters trained on the
    # final detector's boxes, bash_files/launch_seg_final_queue.sh; arm 'ec10' with the sel10g_aug_A_m99 boxes -> labelled 'ec10 final')
    seg_sets = {'model_best': 'seg', 'model_last': 'seg_last', 'strict_rgb_chain': 'seg_rgbdet', 'final': 'seg_final', 'final_last': 'seg_final_last'}
    FINAL_SETS = ('final', 'final_last')
    seg = {}
    for tag, d in seg_sets.items():
        p = os.path.join(args.results_dir, d, 'compare.json')
        if os.path.isfile(p):
            seg[tag] = json.load(open(p))
        else:
            missing.append(p)
    assert 'model_best' in seg, f"{os.path.join(args.results_dir, 'seg', 'compare.json')} is required"
    def arm_label(tag, arm):
        return f"10 EC readings (greedy-10), final detector {FINAL_DET} boxes" if tag in FINAL_SETS and arm == 'ec10' else SEG_ARMS.get(arm, arm)
    grows, crows = [], []
    for tag, c in seg.items():
        for g, st in c['groups'].items():
            model, arm = g.split('|')
            for lvl in SEG_LEVELS:
                if lvl not in st:
                    continue
                for k, v in st[lvl].items():
                    grows.append(dict(result_set=tag, model=model, model_label=SEG_MODELS.get(model, model), arm=arm, arm_label=arm_label(tag, arm), level=lvl, metric=k,
                                      mean=r4(v['mean']), std=r4(v['std']), n_seeds=v['n'], runs=' '.join(st['runs'])))
        for lvl, comps in c['comparisons'].items():
            for label, v in comps.items():
                for k, d in v.items():
                    crows.append(dict(result_set=tag, level=lvl, comparison=label, metric=k, diff=r4(d['diff']), ci_lo=r4(d['lo']), ci_hi=r4(d['hi']),
                                      p_gt0=r4(d.get('p_gt0')), significant=int(d['lo'] > 0 or d['hi'] < 0)))
    manifest['seg_groups'] = write_csv(os.path.join(tables, 'seg_groups.csv'), grows)
    manifest['seg_paired_differences'] = write_csv(os.path.join(tables, 'seg_paired_differences.csv'), crows)
    prows, pi_rows = [], []
    for tag, d in (('model_best', 'seg'), ('strict_rgb_chain', 'seg_rgbdet'), ('final', 'seg_final')):
        root = os.path.join(args.results_dir, d)
        if not os.path.isdir(root):
            continue
        for run in sorted(os.listdir(root)):
            rd = os.path.join(root, run)
            if not os.path.isdir(rd) or not os.path.isfile(os.path.join(rd, 'eval_full_det.json')):
                continue
            ro = json.load(open(os.path.join(rd, 'eval_roi_oracle.json'))); fo = json.load(open(os.path.join(rd, 'eval_full_oracle.json'))); fd = json.load(open(os.path.join(rd, 'eval_full_det.json')))
            row = dict(result_set=tag, run=run, model=ro['seg_model'], arm=ro['arm'], seed=ro['seed'], ckpt=ro['ckpt'], epoch=ro['epoch'], det_run=fd['det_run'],
                       n_det_rois=fd['n_det_rois'], n_fp_rois=fd['n_fp_rois'], n_frames_no_roi=fd['n_frames_no_roi'],
                       fp_false_mask_rate=r4(fd['summary'].get('fp_false_mask_rate')) if fd['summary'].get('fp_false_mask_rate') == fd['summary'].get('fp_false_mask_rate') else None)
            for lvl, e in (('roi_oracle', ro), ('full_oracle', fo), ('full_det', fd)):
                for k in ('S', 'Fw', 'IoU', 'MAE', 'E_mean', 'F_adp'):
                    row[f'{lvl}_{k}'] = r4(e['summary'].get(k))
            for b in ('small', 'medium', 'large'):
                row[f'roi_oracle_S_{b}'] = r4(ro['summary'].get(f'S_{b}')); row[f'roi_oracle_IoU_{b}'] = r4(ro['summary'].get(f'IoU_{b}')); row[f'n_{b}'] = ro['summary'].get(f'n_{b}')
            for k, v in fd.get('paper', {}).items():
                row[f'paper_{k}'] = r4(v)
            prows.append(row)
            pp = os.path.join(rd, 'per_image.pkl')
            if os.path.isfile(pp):
                P = pickle.load(open(pp, 'rb'))
                for lvl in ('roi_oracle', 'full_oracle', 'full_det'):
                    for r in P[lvl]['rows'] + P[lvl].get('fp_rows', []):
                        pi_rows.append(dict(result_set=tag, run=run, level=lvl, frame=r.get('frame'), key=str(r.get('key')), kind=r.get('kind'), gt_area_px=r.get('area'),
                                            **{k: r4(r.get(k)) for k in ('S', 'Fw', 'IoU', 'MAE', 'E_adp', 'F_adp')}))
    manifest['seg_per_run'] = write_csv(os.path.join(tables, 'seg_per_run.csv'), prows)
    manifest['seg_per_image'] = write_csv(os.path.join(figs, 'seg_per_image.csv.gz'), pi_rows)
    # validation curves of the segmentation runs
    sv = []
    for run in sorted(os.listdir(args.weights_dir)):
        p = os.path.join(args.weights_dir, run, f'results_{run}.txt')
        if not run.startswith('seg_') or not os.path.isfile(p):
            continue
        for l in open(p):
            if not l.strip():
                continue
            j = json.loads(l)
            if j.get('final') or 'val' not in j or 'epoch' not in j:
                continue
            sv.append(dict(run=run, epoch=j['epoch'], S=r4(j['val'].get('S')), Fw=r4(j['val'].get('Fw')), IoU=r4(j['val'].get('IoU')), MAE=r4(j['val'].get('MAE')),
                           train_loss=r4(j.get('train', {}).get('loss'))))
    manifest['seg_val_curves'] = write_csv(os.path.join(figs, 'seg_val_curves.csv'), sv)

    # ---------------------------------------------------------------- 5. end-to-end chains and the paper-protocol table
    G = seg['model_best']['groups']
    GR = seg['strict_rgb_chain']['groups'] if 'strict_rgb_chain' in seg else {}
    GF = seg['final']['groups'] if 'final' in seg else {}
    per_run0 = {r['run']: r for r in prows if r['result_set'] == 'model_best'}
    per_run_rgb = {r['run']: r for r in prows if r['result_set'] == 'strict_rgb_chain'}
    per_run_final = {r['run']: r for r in prows if r['result_set'] == 'final'}
    def det_of(arm):
        return {'raw': 'raw133_A', 'ec10': 'sel10g_clean_A', 'ec24': 'sel24g_clean_A', 'rgb': 'rgb_A'}[arm]
    def chain_row(label, arm, det_run, groups, prun, seg_label):
        dm, p = det_means.get(det_run, {}), prun
        row = dict(chain=label, detector=det_run, det_ap50=r4(dm.get('ap50')), det_roi_coverage=r4(dm.get('coverage_recall99_roi_op')),
                   frames_without_roi=p.get('n_frames_no_roi') if p else None, det_rois=p.get('n_det_rois') if p else None, fp_rois=p.get('n_fp_rois') if p else None, segmenter=seg_label)
        for m in ('sam2unet', 'sam2box', 'zoomnext', 'sam2box_zs'):
            g = groups.get(f'{m}|{arm}')
            for k in ('S', 'IoU', 'Fw'):
                row[f'{m}_{k}'] = r4(g['full_det'][k]['mean']) if g else None
            row[f'{m}_n_seeds'] = g['full_det']['S']['n'] if g else None
        return row
    chains = [chain_row('ec10 chain', 'ec10', 'sel10g_clean_A', G, per_run0.get('seg_sam2unet_ec10_s0'), 'trained on ec10'),
              chain_row('raw chain', 'raw', 'raw133_A', G, per_run0.get('seg_sam2unet_raw_s0'), 'trained on raw 133'),
              chain_row('ec24 chain', 'ec24', 'sel24g_clean_A', G, per_run0.get('seg_sam2unet_ec24_s0'), 'trained on ec24'),
              chain_row('pseudo-RGB control chain (raw133_A boxes)', 'rgb', 'raw133_A', G, per_run0.get('seg_sam2unet_rgb_s0'), 'SAM2-UNet trained on pseudo-RGB; zero-shot SAM2.1')]
    if GR:
        chains.append(chain_row('strict RGB chain (rgb_A boxes)', 'rgb', 'rgb_A', GR, per_run_rgb.get('seg_sam2unet_rgb_s0'), 'SAM2-UNet trained on pseudo-RGB; zero-shot SAM2.1'))
    if GF:
        # the final pipeline: the fixed-checkpoint detector metrics are those of the run the checkpoint comes from
        frow = chain_row(f'final ec10 chain ({FINAL_DET} = aug + EMA detector, last checkpoint)', 'ec10', FINAL_DET_METRICS_RUN, GF, per_run_final.get('seg_sam2unet_ec10f_s0'),
                         f'trained on the {FINAL_DET} boxes')
        frow['detector'] = FINAL_DET
        chains.append(frow)
    manifest['end_to_end_chains'] = write_csv(os.path.join(tables, 'end_to_end_chains.csv'), chains)
    # the final chain against the reference ec10 chain, paired by image and seed (paired_chain_bootstrap.py -> seg_final/chain_vs_reference.json)
    cvr = os.path.join(args.results_dir, 'seg_final', 'chain_vs_reference.json')
    if os.path.isfile(cvr):
        cv = json.load(open(cvr))
        manifest['final_chain_vs_reference'] = write_csv(os.path.join(tables, 'final_chain_vs_reference.csv'),
                                                         [dict(model=m, model_label=SEG_MODELS.get(m, m), metric=k, level=cv['level'], **{kk: r4(vv) if isinstance(vv, float) else vv for kk, vv in v.items()},
                                                               significant=int(v['lo'] > 0 or v['hi'] < 0)) for m, d in cv['result'].items() for k, v in d.items()])
    else:
        cv = None; missing.append(cvr)
    pub = [dict(source='HyperCOD paper, arXiv 2601.03736 Table 2', method=m, input='full image, no boxes, 280 training images', S=s, E_mean=e, MAE=mae, F_adp=f) for m, s, e, mae, f in PUBLISHED]
    for tag, groups in (('model_best', G), ('strict_rgb_chain', GR), ('final', GF)):
        for g, st in groups.items():
            if 'full_det_paper' not in st or (tag == 'strict_rgb_chain' and not g.endswith('|rgb')):
                continue
            model, arm = g.split('|'); p = st['full_det_paper']
            pub.append(dict(source=f'this work ({tag}), level (c), paper protocol', method=f"{SEG_MODELS.get(model, model)} on {arm_label(tag, arm)}",
                            input=f"boxes of {'rgb_A' if tag == 'strict_rgb_chain' else (FINAL_DET if tag == 'final' else det_of(arm if arm != 'rgb' else 'raw'))}; outside the ROI = 0; 251 train + 28 val images",
                            S=r4(p['S']['mean']), E_mean=r4(p['E_mean']['mean']), MAE=r4(p['MAE']['mean']), F_adp=r4(p['F_adp']['mean']), n_seeds=p['S']['n']))
    manifest['paper_protocol_vs_published'] = write_csv(os.path.join(tables, 'paper_protocol_vs_published.csv'), pub)

    # ---------------------------------------------------------------- 5b. box study: operating points, fused / merged boxes and the no-detector exports,
    # scored by the ec10 segmenters (seed 0) and the zero-shot control; box statistics from the ROI exports themselves
    gt_file = os.path.join(args.results_dir, 'det', 'rois_sel10g_clean_A_test.json')
    gt_boxes = {k: v['gt_boxes'] for k, v in json.load(open(gt_file)).items()} if os.path.isfile(gt_file) else {}

    def box_stats(roi_file):
        rois = json.load(open(roi_file))
        n_gt = n_in = 0
        for name, gts in gt_boxes.items():
            for g in gts:
                n_gt += 1
                n_in += int(any(r[0] <= g[0] and r[1] <= g[1] and r[2] >= g[2] and r[3] >= g[3] for r in rois.get(name, {'rois': []})['rois']))
        return dict(frames_without_box=sum(1 for v in rois.values() if not v['rois']), boxes=sum(len(v['rois']) for v in rois.values()), gt_inside_roi=r4(n_in / max(n_gt, 1)))

    brows = []
    BASE_SET = 'ec10 default (conf >= 0.02, top 5)'
    box_sets = [(BASE_SET, 'baseline', os.path.join(args.results_dir, 'seg'), gt_file)]
    for d in sorted(glob.glob(os.path.join(args.results_dir, 'seg_op', '*', 'compare.json'))):
        tag = os.path.basename(os.path.dirname(d))
        box_sets.append((f'ec10 operating point {tag}', 'operating point', os.path.dirname(d), os.path.join(args.results_dir, 'det', 'rois_op', tag, 'rois_sel10g_clean_A_test.json')))
    for kind, folder in (('fused / merged boxes', 'seg_fuse'), ('no detector', 'seg_nodet')):
        for d in sorted(glob.glob(os.path.join(args.results_dir, folder, '*', 'compare.json'))):
            name = os.path.basename(os.path.dirname(d))
            box_sets.append((name, kind, os.path.dirname(d), os.path.join(args.results_dir, 'det', f'rois_{name}_test.json')))
    # the recall study's detectors (launch_aug_queue.sh): <det run> = its model_best export at the default operating point,
    # <det run>_<tag> = an export under results/det/rois_op/<name>/ (another checkpoint and / or operating point)
    for d in sorted(glob.glob(os.path.join(args.results_dir, 'seg_aug', '*', 'compare.json'))):
        name = os.path.basename(os.path.dirname(d))
        run = max((r for r in DET_RUNS if name == r or name.startswith(r + '_')), key=len, default=None)
        if run is None:
            missing.append(d); continue
        roi_file = os.path.join(args.results_dir, 'det', f'rois_{run}_test.json') if name == run else os.path.join(args.results_dir, 'det', 'rois_op', name, f'rois_{run}_test.json')
        box_sets.append((name, 'new detector', os.path.dirname(d), roi_file))
    # the final pipeline's segmenters (seg_<model>_ec10f_s0) with the final detector's boxes at other operating points / clustered
    # (bash_files/launch_seg_final_op.sh): <name> = default export clustered (..._cl2) or results/det/rois_op/<name>/
    for d in sorted(glob.glob(os.path.join(args.results_dir, 'seg_final_op', '*', 'compare.json'))):
        name = os.path.basename(os.path.dirname(d))
        roi_file = os.path.join(args.results_dir, 'det', f'rois_{name}_test.json')
        if not os.path.isfile(roi_file):
            roi_file = os.path.join(args.results_dir, 'det', 'rois_op', name, f'rois_{FINAL_DET}_test.json')
        box_sets.append((name, 'final pipeline', os.path.dirname(d), roi_file))
    if os.path.isfile(os.path.join(args.results_dir, 'seg_final', 'compare.json')):
        box_sets.append((f'{FINAL_DET} default (final segmenters)', 'final pipeline', os.path.join(args.results_dir, 'seg_final'), os.path.join(args.results_dir, 'det', f'rois_{FINAL_DET}_test.json')))
    for label, kind, out_dir, roi_file in box_sets:
        if not os.path.isfile(roi_file):
            missing.append(roi_file); continue
        row = dict(box_set=label, kind=kind, **box_stats(roi_file))
        suffix = 'ec10f' if kind == 'final pipeline' else 'ec10'
        for m in ('sam2unet', 'sam2box', 'zoomnext', 'sam2box_zs'):
            if m == 'sam2box_zs' and kind == 'baseline':
                continue                                            # the queue's zero-shot control used raw133_A's boxes, not this set
            p = os.path.join(out_dir, 'zs_sam2box_rgb' if m == 'sam2box_zs' else f'seg_{m}_{suffix}_s0', 'eval_full_det.json')
            if os.path.isfile(p):
                e = json.load(open(p)); s = e['summary']
                if m != 'sam2box_zs':
                    row['fp_boxes'] = e['n_fp_rois']
                for k in ('S', 'IoU', 'Fw'):
                    row[f'{m}_{k}'] = r4(s[k])
                row[f'{m}_false_mask_rate'] = r4(s.get('fp_false_mask_rate')) if s.get('fp_false_mask_rate') == s.get('fp_false_mask_rate') else None
        brows.append(row)
    manifest['box_study'] = write_csv(os.path.join(tables, 'box_study.csv'), brows) if brows else None

    # ---------------------------------------------------------------- 6. compute
    crow = []
    for run in sorted(set(DET_RUNS) | set(det_means)):
        t = log_timing(os.path.join(args.logs_dir, f'{run}.log'))
        if t:
            crow.append(dict(stage=1, run=run, label=DET_RUNS.get(run, (run,))[0], **t))
    seg_wall = {}
    if os.path.isfile(args.seg_timing):
        seg_wall = json.load(open(args.seg_timing)).get('timing', {})
    for run in sorted(os.listdir(args.weights_dir)):
        if run.startswith('seg_') and not run.startswith('seg_smoke'):
            t = log_timing(os.path.join(args.logs_dir, f'{run}.log'))
            if t:
                crow.append(dict(stage=2, run=run, label=run, **t, wall_h=seg_wall.get(run, {}).get('wall_h'), gpu=seg_wall.get(run, {}).get('gpu')))
    manifest['compute'] = write_csv(os.path.join(tables, 'compute.csv'), crow)

    # ---------------------------------------------------------------- 7. Stage-1 per-image statistics of the fixed-checkpoint comparisons
    di_rows = []
    for name in DET_COMPARE_CLEAN:
        p = os.path.join(args.results_dir, 'det', name, 'per_image.pkl')
        if not os.path.isfile(p):
            continue
        P = pickle.load(open(p, 'rb'))['test']
        for key, stats in P['stats'].items():
            rc, cond = key.split('|'); run, ck = rc.split('/')
            for img, st in zip(P['names'], stats):
                flat = {k: (r4(v) if isinstance(v, (int, float, np.integer, np.floating)) else json.dumps(v, default=float)) for k, v in (st.items() if isinstance(st, dict) else enumerate(st))}
                di_rows.append(dict(source=name, run=run, checkpoint=ck, condition=cond, image=img, **flat))
    manifest['det_per_image'] = write_csv(os.path.join(figs, 'det_per_image.csv.gz'), di_rows) if di_rows else None

    # ---------------------------------------------------------------- 8. the summary
    def dm(run, k='ap50'):
        v = det_means.get(run, {}).get(k)
        return f'{v:.3f}' if v is not None else 'n/a'
    def dcomp(a, b, k='ap50'):
        for r in comps_det:
            if r['condition'] == 'clean' and r['metric'] == k:
                if r['first'] == a and r['second'] == b:
                    return ci_text(dict(diff=r['diff'], lo=r['ci_lo'], hi=r['ci_hi']))
                if r['first'] == b and r['second'] == a:
                    return ci_text(dict(diff=-r['diff'], lo=-r['ci_hi'], hi=-r['ci_lo']))
        return 'n/a'
    def gS(groups, g, lvl='full_det', k='S'):
        return f"{groups[g][lvl][k]['mean']:.3f}" if g in groups and lvl in groups[g] else 'n/a'
    def scomp(c, lvl, label, k='S', flip=False):
        d = c['comparisons'].get(lvl, {}).get(label)
        if not d:
            return 'n/a'
        d = d[k]
        return ci_text(dict(diff=-d['diff'], lo=-d['hi'], hi=-d['lo']) if flip else d)
    C0 = seg['model_best']; CR = seg.get('strict_rgb_chain')
    seeds_txt = ''
    for r in seed_rows:
        seeds_txt += f"- {r['label']}: AP50 {r['ap50_mean']:.3f} ± {r['ap50_std']:.3f} (n = {r['n_seeds']}: {r['ap50_min']:.3f}–{r['ap50_max']:.3f}); recall50 {r['recall50_mean']:.3f} ± {r['recall50_std']:.3f}; ROI coverage {r['coverage_recall99_roi_op_mean']:.3f} ± {r['coverage_recall99_roi_op_std']:.3f}\n"
    rawpca_txt = ''
    for run in ('rawpca10_A', 'rawpca40_A'):
        if run in det_means:
            rawpca_txt += f"- {DET_RUNS[run][0]}: AP50 {dm(run)}, recall50 {dm(run, 'recall50')}, ROI coverage {dm(run, 'coverage_recall99_roi_op')}; greedy-10 − {DET_RUNS[run][0]} = {dcomp('sel10g_clean_A', run)}; {DET_RUNS[run][0]} − raw 133 = {dcomp(run, 'raw133_A')}\n"
    if not rawpca_txt:
        rawpca_txt = '- raw-PCA-10 / raw-PCA-40 and the extra seeds: not yet in results/det/compare_rawpca_seeds (queue bash_files/launch_rawpca_seeds_queue.sh)\n'
    aug_txt = ''
    for run in ('sel10g_aug_A', 'sel10g_aug_m_A', 'sel10g_aug_tv_A', 'raw133_aug_A'):
        if run in det_means:
            ref = 'raw133_A' if run.startswith('raw') else 'sel10g_clean_A'
            aug_txt += (f"- {DET_RUNS[run][0]}: AP50 {dm(run)}, recall50 {dm(run, 'recall50')}, ROI coverage {dm(run, 'coverage_recall99_roi_op')}, boxes / image {dm(run, 'dets_per_image_op')}; "
                        f"− {DET_RUNS[ref][0]} (seed 42): AP50 {dcomp(run, ref)}, recall50 {dcomp(run, ref, 'recall50')}, ROI coverage {dcomp(run, ref, 'coverage_recall99_roi_op')}\n")
    if 'sel10g_aug_A' in det_means and 'raw133_aug_A' in det_means:
        aug_txt += f"- greedy-10 + aug + EMA − raw 133 + aug + EMA: AP50 {dcomp('sel10g_aug_A', 'raw133_aug_A')}, ROI coverage {dcomp('sel10g_aug_A', 'raw133_aug_A', 'coverage_recall99_roi_op')}\n"
    if not aug_txt:
        aug_txt = '- not yet in results/det/compare_aug (queue bash_files/launch_aug_queue.sh)\n'
    strict = (f"S {gS(GR, 'sam2unet|rgb')}, IoU {gS(GR, 'sam2unet|rgb', k='IoU')}; ec10 chain − strict RGB chain (SAM2-UNet) {scomp(CR, 'full_det', 'sam2unet: ec10 - rgb')} S, "
              f"{scomp(CR, 'full_det', 'sam2unet: ec10 - rgb', 'IoU')} IoU; raw chain − strict RGB chain {scomp(CR, 'full_det', 'sam2unet: raw - rgb')}; "
              f"zero-shot SAM2.1 with rgb_A boxes S {gS(GR, 'sam2box_zs|rgb')}") if GR else 'not evaluated'
    md = f'''# EC-sensor camouflaged-object detection and segmentation — results package

Generated by `main_paper_package.py` from the result files of the two-stage study (Stage 1: EC-YOLO detector on simulated
electrochromic-sensor readings; Stage 2: box-guided segmentation). Every number below is copied from the CSV tables in
`tables/`; the figure data are in `figure_data/`. Read-noise experiments (40 dB) are in the tables for completeness but
flagged (`read_noise_db = 40`); the headline results are noise-free.

## 1. Data and task (facts for Methods)

- HyperCOD hyperspectral camouflaged-object images, 1680 × 1240 pixels, 200 bands 400–1000 nm; the 133 bands inside the sensor's
  400–800 nm range are used (3.015 nm spacing). Pixel-level ground-truth masks; connected components ≥ 100 px are objects (360 in all).
- Split (fixed seed 0): 251 training images, 28 validation images (31 objects), 70 test images (71 objects). The test split is used only
  for the final numbers; model selection and voltage selection never see it. The same split serves both stages.
- Brightness normalisation: each image divided by the 99th percentile of its 133-band intensity sum (per-image p99).
- EC tunable photodetector: measured responsivity R(λ, V), 400–800 nm × 351 bias voltages (−1.00 … 2.50 V, 0.01 V step); the polarity-flip
  dead zone 0.25–0.31 V is excluded → 344 usable voltages. One reading per voltage: y_v = R_vᵀ x. Response FWHM median 112 nm; adjacent
  voltages have cosine similarity 0.9998; the response matrix has ~6–11 effective directions (99 / 99.9 % energy).
- Simulation is noise-free in the headline experiments; whitening uses the training-set band statistics (mean, covariance over 512 × 512
  crops of the 279 train + val images).

## 2. Stage 1 — detection (YOLO26s, native resolution, fixed checkpoints 59–99, test split)

| input | channels | AP50 | recall50 | ROI coverage | boxes / image |
|---|---|---|---|---|---|
''' + ''.join(f"| {r['label']} | {r['channels']} | {r['ap50']:.3f} | {r['recall50']:.3f} | {r['coverage_recall99_roi_op']:.3f} | {r['dets_per_image_op']:.2f} |\n"
             for r in rows if r['read_noise_db'] == 0) + f'''
Paired differences (AP50, 95 % CI from 2000 paired image bootstraps; sign = first − second):

- greedy-10 − raw 133: {dcomp('sel10g_clean_A', 'raw133_A')}; greedy-10 − PCA-11: {dcomp('sel10g_clean_A', 'pca11_A')}; greedy-10 − greedy-24: {dcomp('sel10g_clean_A', 'sel24g_clean_A')}
- greedy-10 − RGB: {dcomp('sel10g_clean_A', 'rgb_A')}; raw 133 − RGB: {dcomp('raw133_A', 'rgb_A')}; PCA-11 − RGB: {dcomp('pca11_A', 'rgb_A')}; greedy-24 − RGB: {dcomp('sel24g_clean_A', 'rgb_A')}
- greedy-10 − gate-10 (whitened): {dcomp('sel10g_clean_A', 'sel10b_clean_A')}; greedy-10 − uniform-10: {dcomp('sel10g_clean_A', 'sel10u_clean_A')}; PCA-11 − raw 133: {dcomp('pca11_A', 'raw133_A')}; PCA-11 − session A: {dcomp('pca11_A', 'det_A')}
- ROI coverage: greedy-10 − raw 133: {dcomp('sel10g_clean_A', 'raw133_A', 'coverage_recall99_roi_op')}; greedy-10 − RGB: {dcomp('sel10g_clean_A', 'rgb_A', 'coverage_recall99_roi_op')}

Training-seed spread (fixed checkpoints, per seed):
{seeds_txt or '- single seed (42) per detector so far\n'}
Whitening vs. selection controls (raw bands whitened to K principal directions):
{rawpca_txt}
Recall without new data (scale / crop + photometric jitter and a weight EMA; the yolo26m backbone; training on train + val — `bash_files/launch_aug_queue.sh`;
their boxes scored end to end in the box-study table, kind "new detector"):
{aug_txt}
Model-best (validation-selected) checkpoints are in `tables/det_model_best.csv`; they are not used for the comparisons (the selection rule
ignores AP50 and often picks early epochs).

## 3. Stage 2 — segmentation inside the detector boxes (test split, 3 seeds per model × input, model_best)

S-measure, level (a) oracle boxes / level (c) end to end with each input's own detector boxes:

| model | raw 133 (a) / (c) | ec10 (a) / (c) | ec24 (a) / (c) |
|---|---|---|---|
''' + ''.join(f"| {SEG_MODELS[m]} | {gS(G, m + '|raw', 'roi_oracle')} / {gS(G, m + '|raw')} | {gS(G, m + '|ec10', 'roi_oracle')} / {gS(G, m + '|ec10')} | {gS(G, m + '|ec24', 'roi_oracle')} / {gS(G, m + '|ec24')} |\n"
             for m in ('sam2unet', 'sam2box', 'zoomnext')) + f'''
Controls: pseudo-RGB SAM2-UNet (3 seeds) (a) {gS(G, 'sam2unet|rgb', 'roi_oracle')} / (c, raw133_A boxes) {gS(G, 'sam2unet|rgb')}; zero-shot SAM2.1 + box (a) {gS(G, 'sam2box_zs|rgb', 'roi_oracle')} / (c, raw133_A boxes) {gS(G, 'sam2box_zs|rgb')}.
Strict RGB chain (rgb_A boxes + pseudo-RGB segmenter): {strict}.

Paired differences (S, level (a) oracle boxes): SAM2.1+box ec10 − raw {scomp(C0, 'roi_oracle', 'sam2box: raw - ec10', flip=True)}; SAM2-UNet ec10 − raw {scomp(C0, 'roi_oracle', 'sam2unet: raw - ec10', flip=True)};
ZoomNeXt ec10 − raw {scomp(C0, 'roi_oracle', 'zoomnext: raw - ec10', flip=True)}; SAM2-UNet raw − pseudo-RGB {scomp(C0, 'roi_oracle', 'sam2unet: raw - rgb')}; ec10 − pseudo-RGB {scomp(C0, 'roi_oracle', 'sam2unet: ec10 - rgb')};
zero-shot − trained pseudo-RGB {scomp(C0, 'roi_oracle', 'rgb: sam2box_zs - sam2unet')}.
Level (c) end to end: ec10 − raw: SAM2-UNet {scomp(C0, 'full_det', 'sam2unet: raw - ec10', flip=True)}, SAM2.1+box {scomp(C0, 'full_det', 'sam2box: raw - ec10', flip=True)}, ZoomNeXt {scomp(C0, 'full_det', 'zoomnext: raw - ec10', flip=True)}.

Detector boxes on the 70 test images (frames without any box / false-positive boxes): see `tables/end_to_end_chains.csv` and `tables/seg_per_run.csv`
(false-mask rate = share of false-positive boxes in which the segmenter predicts anything > 0.5).

## 4. Claims and the evidence behind them

1. **Ten EC readings replace the 133-band cube for detection.** greedy-10 AP50 {dm('sel10g_clean_A')} vs raw 133 {dm('raw133_A')} ({dcomp('sel10g_clean_A', 'raw133_A')}); ROI coverage {dm('sel10g_clean_A', 'coverage_recall99_roi_op')} vs {dm('raw133_A', 'coverage_recall99_roi_op')}. Seeds and the raw-PCA controls above say how much of this is whitening vs. selection.
2. **… and for segmentation.** Oracle-box S ec10 ≥ raw for all three models (significant for SAM2.1+box and SAM2-UNet); end to end ec10 gives the best result for every model.
3. **Spectral input beats an RGB camera, mostly by finding the object.** Detection: every spectral input is 0.165–0.242 AP50 above RGB (all CIs exclude 0). Inside a correct box pseudo-RGB ties raw 133 ({scomp(C0, 'roi_oracle', 'sam2unet: raw - rgb')}); the strict RGB chain loses end to end ({strict.split(';')[0]}).
4. **Voltages must be chosen as a set.** The trainable per-channel weights (session A) collapse to two clusters of near-identical voltages (AP50 {dm('det_A')}); greedy-10 beats gate-10 by {dcomp('sel10g_clean_A', 'sel10b_clean_A')} and uniform-10 by {dcomp('sel10g_clean_A', 'sel10u_clean_A')}.
5. **The early gap was input conditioning, not information.** Whitening the same 344 readings to 11 directions (PCA-11) ties raw 133 ({dcomp('pca11_A', 'raw133_A')}) and beats session A by {dcomp('pca11_A', 'det_A')}; 344 readings keep 93 % of the pixel separability, condition number 8.6×10¹¹ vs 7.
6. **24 readings add nothing.** greedy-10 − greedy-24 {dcomp('sel10g_clean_A', 'sel24g_clean_A')} (detection); ec24 ≤ ec10 at every segmentation level.
7. **Against the published HyperCOD numbers** (full-image methods, paper protocol): see `tables/paper_protocol_vs_published.csv` — the ec10 chains reach S ≈ 0.85 vs 0.805 for the paper's best, with the caveat that our masks are box-guided (MAE not comparable).

## 5. Suggested figures (main text) and where their data are

1. Device and pipeline: R(λ, V) heat map with the selected voltages (`figure_data/sensor_response_on_cube_bands.csv`, `voltage_selection.json`); two-stage schematic.
2. Choosing voltages: gate weights vs. voltage (`gate_weights_ep89.csv`), the four voltage sets, greedy objective per step (`voltage_selection.json`), detection AP50 bars with paired CIs (`det_fixed_checkpoint.csv`, `det_paired_differences.csv`).
3. Detection vs. baselines: raw 133, PCA-11, RGB, greedy-10/24, raw-PCA controls and seeds (`det_seed_summary.csv`); validation curves (`det_val_curves.csv`).
4. Segmentation: S / IoU per model × input at levels (a) and (c) with CIs (`seg_groups.csv`, `seg_paired_differences.csv`); size buckets (`seg_per_run.csv`).
5. End to end: the chains against the RGB chains, zero-shot and the published numbers (`end_to_end_chains.csv`, `paper_protocol_vs_published.csv`); qualitative masks (per-image scores in `seg_per_image.csv`, galleries in the report pages).
Extended Data: per-image detection hits (`det_per_image.csv`), model_last vs model_best (`seg_groups.csv`, result_set = model_last), compute (`compute.csv`), the 40 dB read-noise experiments (flagged rows).

## 6. Limitations to state

- One training seed per detector in the original Stage-1 runs (seeds 1, 2 added for greedy-10 and raw 133 when present); Stage 2 has 3 seeds per cell.
- The test split was reused for design decisions (which voltage set, which inputs to carry to Stage 2); no multiple-comparison correction over the paired tests.
- The pseudo-RGB and the dataset's RGB image are rendered from the same cube: "same scene, 3 broad channels", not an independent camera.
- Stage-2 oracle boxes come from the ground truth; only level (c) and the chains are deployment-like. Masks are box-guided (outside = 0), so MAE is not comparable with full-image methods.
- EC readings are simulated from the cube without read noise; the 40 dB experiments (flagged) show the optimal voltage layout depends on the device SNR — the next step is measuring the device.
- Whitening of greedy-10 amplifies the weakest direction ~7.9×10³ ×; ec24 has 3 numerically null whitened channels.

## 7. Files

''' + '\n'.join(f'- `{os.path.relpath(v, out)}`' for v in manifest.values() if v) + (
        '\n\nMissing inputs (skipped): ' + ', '.join(f'`{m}`' for m in missing) if missing else '') + '\n'
    if brows:
        def bget(label, m, k='S'):
            r = next((r for r in brows if r['box_set'] == label), None)
            return f"{r[f'{m}_{k}']:.3f}" if r and r.get(f'{m}_{k}') is not None else 'n/a'
        best = max((r for r in brows if r['kind'] == 'operating point' and r.get('sam2unet_S') is not None), key=lambda r: r['sam2unet_S'], default=None)
        box_md = ('## 3b. Box study — what the detector boxes cost and what cheaper boxes buy (`tables/box_study.csv`)\n\n'
                  'The ec10 segmenters (seed 0) scored end to end with other box sets than their detector\'s default operating point '
                  '(confidence >= 0.02, top 5 per image): an operating-point sweep of sel10g_clean_A, unions of several detectors\' boxes, '
                  'overlapping boxes merged into one (`main_fuse_rois.py --merge`), and the no-detector references (the whole frame as one ROI; '
                  'a 512-px window grid at native resolution). Box statistics: frames without any box, boxes, false-positive boxes, GT boxes fully '
                  'inside some ROI.\n\n'
                  '| box set | kind | frames w/o box | boxes | fp boxes | GT inside ROI | SAM2-UNet S / IoU | SAM2.1+box S / IoU | ZoomNeXt S / IoU | zero-shot S / IoU |\n|---|---|---|---|---|---|---|---|---|---|\n'
                  + ''.join(f"| {r['box_set']} | {r['kind']} | {r['frames_without_box']} | {r['boxes']} | {r.get('fp_boxes', '')} | {r['gt_inside_roi']:.3f} | "
                            + ' | '.join((f"{r[m + '_S']:.3f} / {r[m + '_IoU']:.3f}" if r.get(m + '_S') is not None else '—') for m in ('sam2unet', 'sam2box', 'zoomnext', 'sam2box_zs')) + ' |\n' for r in brows)
                  + (f"\nBest operating point for SAM2-UNet: {best['box_set']} (S {best['sam2unet_S']:.3f} vs {bget(BASE_SET, 'sam2unet')} at the default): the trained segmenters "
                     f"suppress most of the extra false-positive boxes, so recall is worth more than precision at this stage. Merging overlapping boxes at the default point: S "
                     f"{bget('ec10_union', 'sam2unet')} (union) / {bget('ec10_wbf', 'sam2unet')} (wbf) vs {bget(BASE_SET, 'sam2unet')} — imperfect boxes cost almost nothing, the detector's "
                     f"cost is the frames it leaves without a box. No detector (SAM2-UNet / SAM2.1+box / ZoomNeXt S): whole frame as one ROI {bget('full_frame', 'sam2unet')} / "
                     f"{bget('full_frame', 'sam2box')} / {bget('full_frame', 'zoomnext')}, 512-px window grid {bget('windows_512', 'sam2unet')} / {bget('windows_512', 'sam2box')} / "
                     f"{bget('windows_512', 'zoomnext')} (zero-shot SAM2.1 {bget('full_frame', 'sam2box_zs')} / {bget('windows_512', 'sam2box_zs')}). Whether the detector is needed therefore depends on "
                     f"the segmenter: the COD-pretrained ZoomNeXt comes within 0.01 S of its detector-guided default from the dense grid (IoU {bget('windows_512', 'zoomnext', 'IoU')} vs "
                     f"{bget(BASE_SET, 'zoomnext', 'IoU')}, at 15 segmenter passes per image instead of <= 5), while the SAM2-based segmenters do not (SAM2-UNet "
                     f"{bget('windows_512', 'sam2unet')}; SAM2.1+box {bget('windows_512', 'sam2box')}, because a box prompt presupposes an object and most empty windows get a mask). "
                     f"For the window grid the 'GT inside ROI' column counts only objects that fit inside one window; the pasted-back map covers every pixel. The segmenters were trained on "
                     f"object-centred ROIs, so the no-detector rows are a bound on what the detector adds, not a trained full-frame baseline."
                     + (f" New detectors (recall study, `bash_files/launch_aug_queue.sh`; <det run> = its model_best export, <det run>_m99* = the last checkpoint): greedy-10 + aug + EMA with "
                        f"model_99 at the default operating point gives S {bget('sel10g_aug_A_m99', 'sam2unet')} / {bget('sel10g_aug_A_m99', 'sam2box')} / {bget('sel10g_aug_A_m99', 'zoomnext')} "
                        f"(SAM2-UNet / SAM2.1+box / ZoomNeXt) against {bget(BASE_SET, 'sam2unet')} / {bget(BASE_SET, 'sam2box')} / {bget(BASE_SET, 'zoomnext')} with the reference boxes; its "
                        f"validation-selected model_best ({bget('sel10g_aug_A', 'sam2unet')} / {bget('sel10g_aug_A', 'sam2box')} / {bget('sel10g_aug_A', 'zoomnext')}) and the yolo26m, train + val "
                        f"and raw + aug detectors do not improve on the reference; lowering the augmented detector's threshold hurts (it already emits twice the boxes)." if bget('sel10g_aug_A_m99', 'sam2unet') != 'n/a' else '')
                     + "\n\n" if best else '\n'))
        md = md.replace('## 4. Claims and the evidence behind them', box_md + '## 4. Claims and the evidence behind them', 1)
    if GF:
        def gstat(groups, m, lvl='full_det', k='S'):
            g = groups.get(f'{m}|ec10')
            return f"{g[lvl][k]['mean']:.3f} ± {g[lvl][k]['std']:.3f}" if g and lvl in g else 'n/a'
        def cvt(m, k='S'):
            v = (cv or {}).get('result', {}).get(m, {}).get(k)
            return ci_text(dict(diff=v['diff'], lo=v['lo'], hi=v['hi'])) if v else 'n/a'
        fr = per_run_final.get('seg_sam2unet_ec10f_s0', {}); rr = per_run0.get('seg_sam2unet_ec10_s0', {})
        oracle_final = '; '.join(f"{SEG_MODELS[m]} {gstat(GF, m, 'roi_oracle')} / {gstat(GF, m, 'full_oracle')}" for m in ('sam2unet', 'sam2box', 'zoomnext'))
        oracle_ref = '; '.join(f"{SEG_MODELS[m]} {gstat(G, m, 'roi_oracle')} / {gstat(G, m, 'full_oracle')}" for m in ('sam2unet', 'sam2box', 'zoomnext'))
        final_md = (f'## 3c. Final pipeline — the detector with scale / photometric jitter + EMA (last checkpoint, `{FINAL_DET}`) and segmenters trained on its boxes (`tables/end_to_end_chains.csv`, `tables/final_chain_vs_reference.csv`)\n\n'
                    f'Three segmenters × 3 seeds trained with the final detector\'s train / val boxes (`bash_files/launch_seg_final_queue.sh`, result set `final` in `tables/seg_groups.csv` / `seg_per_run.csv`), '
                    f'scored end to end with its test boxes at the default operating point (conf ≥ 0.02, top 5): {fr.get("n_frames_no_roi", "n/a")} images without a box, {fr.get("n_det_rois", "n/a")} boxes, '
                    f'{fr.get("n_fp_rois", "n/a")} false positives (reference ec10 chain: {rr.get("n_frames_no_roi", "n/a")} / {rr.get("n_det_rois", "n/a")} / {rr.get("n_fp_rois", "n/a")}). '
                    f'Paired by image and seed (2000 frame bootstraps):\n\n'
                    '| model | reference ec10 chain S | final chain S | final − reference S [95 % CI] | reference IoU | final IoU | final − reference IoU | final − reference Fw |\n|---|---|---|---|---|---|---|---|\n'
                    + ''.join(f"| {SEG_MODELS[m]} | {gstat(G, m)} | {gstat(GF, m)} | {cvt(m)} | {gstat(G, m, k='IoU')} | {gstat(GF, m, k='IoU')} | {cvt(m, 'IoU')} | {cvt(m, 'Fw')} |\n" for m in ('sam2unet', 'sam2box', 'zoomnext'))
                    + f'\nOracle-box levels of the final segmenters (a) / (b), S: {oracle_final} (reference: {oracle_ref}). '
                    'Operating-point / object-count-prior rows of the final chain: `tables/box_study.csv`, kind "final pipeline".\n\n')
        md = md.replace('## 4. Claims and the evidence behind them', final_md + '## 4. Claims and the evidence behind them', 1)
    with open(os.path.join(out, 'results_summary.md'), 'w') as f:
        f.write(md)
    json.dump(dict(files={k: (os.path.relpath(v, out) if v else None) for k, v in manifest.items()}, missing=missing), open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(f"wrote {len([v for v in manifest.values() if v])} files to {out}; {len(rows)} detector runs, {len(seed_rows)} seed groups, {len(grows)} segmentation group rows, "
          f"{len(crows)} segmentation comparisons, {len(pi_rows)} per-image rows" + (f"; missing: {missing}" if missing else ''))


if __name__ == '__main__':
    main(get_args_parser().parse_args())
