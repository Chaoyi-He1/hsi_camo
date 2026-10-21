import json

import main_fuse_rois
import main_window_rois


def test_fuse_frame_unions_boxes_drops_duplicates_and_caps():
    a = {'rois': [[0, 0, 20, 20, 0.9], [40, 40, 60, 60, 0.3]], 'boxes': [[5, 5, 15, 15, 0.9], [45, 45, 55, 55, 0.3]], 'gt_boxes': [[6, 6, 14, 14]]}
    b = {'rois': [[1, 1, 21, 21, 0.8], [70, 70, 90, 90, 0.5]], 'boxes': [[5, 5, 15, 16, 0.8], [75, 75, 85, 85, 0.5]], 'gt_boxes': [[6, 6, 14, 14]]}
    f = main_fuse_rois.fuse_frame([a, b], iou_dedup=0.7)
    # the near-duplicate of the 0.9 box (IoU 0.91) is dropped, the rest is the union sorted by confidence
    assert [r[4] for r in f['rois']] == [0.9, 0.5, 0.3] and f['boxes'][1][:4] == [75, 75, 85, 85] and f['gt_boxes'] == [[6, 6, 14, 14]]
    assert main_fuse_rois.fuse_frame([a, b], iou_dedup=1.0)['rois'][1][4] == 0.8              # 1.0 keeps everything
    assert [r[4] for r in main_fuse_rois.fuse_frame([a, b], iou_dedup=0.7, max_rois=2)['rois']] == [0.9, 0.5]
    assert main_fuse_rois.fuse_frame([{'rois': [], 'boxes': [], 'gt_boxes': [[1, 1, 2, 2]]}], 0.7) == {'rois': [], 'boxes': [], 'gt_boxes': [[1, 1, 2, 2]]}


def test_fuse_frame_merges_overlapping_boxes_into_one_regrown_roi():
    # two half boxes of one object (IoU 0.6) and a far box; merge = union -> the enclosing box, ROI re-grown (x1.5, >= 20 px, clipped)
    a = {'rois': [[0, 0, 30, 30, 0.9], [50, 50, 80, 80, 0.4]], 'boxes': [[10, 10, 20, 20, 0.9], [60, 60, 70, 70, 0.4]], 'gt_boxes': []}
    b = {'rois': [[2, 0, 32, 30, 0.8]], 'boxes': [[12, 10, 22, 20, 0.8]], 'gt_boxes': []}
    f = main_fuse_rois.fuse_frame([a, b], merge='union', iou_merge=0.5, roi_margin=1.5, roi_min=20, frame_hw=(100, 100))
    assert f['boxes'][0] == [10.0, 10.0, 22.0, 20.0, 0.9]                                       # enclosing box, max conf
    assert f['rois'][0] == [6.0, 5.0, 26.0, 25.0, 0.9]                                           # centre (16, 15), 12 x 10 -> both sides lifted to the 20 px minimum
    assert f['boxes'][1][:4] == [60, 60, 70, 70] and f['rois'][1] == [50, 50, 80, 80, 0.4]      # a lone box keeps its exported ROI
    w = main_fuse_rois.fuse_frame([a, b], merge='wbf', iou_merge=0.5, roi_min=20, frame_hw=(100, 100))['boxes'][0]
    assert abs(w[0] - (10 * 0.9 + 12 * 0.8) / 1.7) < 1e-9 and w[4] == 0.9                      # confidence-weighted coordinates
    # below the merge threshold nothing merges and the ROIs stay as exported
    assert main_fuse_rois.fuse_frame([a, b], merge='union', iou_merge=0.9)['rois'][0] == [0, 0, 30, 30, 0.9]


def test_cluster_merge_keeps_the_n_most_confident_object_clusters():
    '''The "at most N objects per image" prior: overlapping / adjacent boxes form one object (enclosing box), the N most confident
    clusters survive, the rest are dropped as false positives.'''
    a = {'rois': [[0, 0, 40, 40, 0.9], [8, 0, 48, 40, 0.3], [60, 60, 80, 80, 0.5], [0, 60, 20, 80, 0.4], [20, 1, 30, 11, 0.2]],
         'boxes': [[10, 10, 20, 20, 0.9], [15, 12, 28, 22, 0.3], [65, 65, 75, 75, 0.5], [2, 65, 12, 75, 0.4], [21, 2, 29, 10, 0.2]], 'gt_boxes': []}
    assert main_fuse_rois.linked([10, 10, 20, 20], [15, 12, 28, 22]) and main_fuse_rois.linked([10, 10, 20, 20], [20, 10, 30, 20])   # overlap, touch
    assert not main_fuse_rois.linked([10, 10, 20, 20], [22, 10, 32, 20]) and main_fuse_rois.linked([10, 10, 20, 20], [22, 10, 32, 20], link_gap=0.2)   # 2 px gap, side 10
    f = main_fuse_rois.fuse_frame([a], merge='cluster', max_rois=2, roi_min=20, frame_hw=(100, 100))
    # clusters: {0.9, 0.3} overlap -> enclosing box; {0.5}; {0.4}; {0.2} touches neither (gap 1 px) -> four clusters, the two most confident kept
    assert f['boxes'][0] == [10.0, 10.0, 28.0, 22.0, 0.9] and f['boxes'][1][:4] == [65, 65, 75, 75] and len(f['boxes']) == 2
    assert f['rois'][0] == [5.5, 6.0, 32.5, 26.0, 0.9] and f['rois'][1] == [60, 60, 80, 80, 0.5]      # merged box re-grown: centre (19, 16), 18 x 12 -> 27 x 20 (height lifted to the minimum); a lone box keeps its ROI
    g = main_fuse_rois.fuse_frame([a], merge='cluster', max_rois=2, link_gap=0.2, roi_min=20, frame_hw=(100, 100))
    assert g['boxes'][0][:4] == [10.0, 2.0, 29.0, 22.0]                                            # the 0.2 box (1 px away) joins the first cluster
    assert len(main_fuse_rois.fuse_frame([a], merge='cluster')['boxes']) == 4                           # max_rois 0 keeps every cluster
    assert main_fuse_rois.fuse_frame([{'rois': [], 'boxes': [], 'gt_boxes': []}], merge='cluster', max_rois=2)['boxes'] == []


def test_main_writes_the_fused_export_over_the_union_of_frames(tmp_path, capsys):
    e1 = {'7': {'rois': [[0, 0, 20, 20, 0.9]], 'boxes': [[5, 5, 15, 15, 0.9]], 'gt_boxes': [[6, 6, 14, 14]]}, '8': {'rois': [], 'boxes': [], 'gt_boxes': [[30, 30, 40, 40]]}}
    e2 = {'8': {'rois': [[28, 28, 42, 42, 0.4]], 'boxes': [[31, 31, 39, 39, 0.4]], 'gt_boxes': [[30, 30, 40, 40]]}, '9': {'rois': [], 'boxes': [], 'gt_boxes': []}}
    (tmp_path / 'a.json').write_text(json.dumps(e1)); (tmp_path / 'b.json').write_text(json.dumps(e2))
    main_fuse_rois.main(main_fuse_rois.get_args_parser().parse_args(['--inputs', str(tmp_path / 'a.json'), str(tmp_path / 'b.json'), '--name', 'fz', '--out_dir', str(tmp_path)]))
    out = json.loads((tmp_path / 'rois_fz_test.json').read_text())
    assert set(out) == {'7', '8', '9'} and len(out['8']['rois']) == 1 and out['9']['rois'] == []
    msg = capsys.readouterr().out
    assert '2 exports, 2 boxes in -> 2 out over 3 frames, 1 frames without any box, 2/2 GT boxes fully inside an ROI (1.000)' in msg


def test_window_rois_cover_the_frame_and_the_frame_mode_is_one_roi(tmp_path):
    grid = main_window_rois.window_grid(1680, 1240, 512, 384)
    assert len(grid) == 15 and grid[0] == (0, 0) and grid[-1] == (728, 1168)                  # 3 columns (0, 384, 728) x 5 rows (…, 1168 = 1680 - 512)
    assert all(x0 + 512 <= 1240 and y0 + 512 <= 1680 for x0, y0 in grid)
    assert main_window_rois.frame_rois('frame', 1680, 1240) == [[0.0, 0.0, 1240.0, 1680.0, 1.0]]
    like = {'7': {'rois': [], 'boxes': [], 'gt_boxes': [[1, 1, 2, 2]]}, '8': {'rois': [], 'boxes': [], 'gt_boxes': []}}
    (tmp_path / 'like.json').write_text(json.dumps(like))
    main_window_rois.main(main_window_rois.get_args_parser().parse_args(['--like', str(tmp_path / 'like.json'), '--mode', 'windows', '--name', 'win', '--out_dir', str(tmp_path)]))
    out = json.loads((tmp_path / 'rois_win_test.json').read_text())
    assert set(out) == {'7', '8'} and len(out['7']['rois']) == 15 and out['7']['rois'] == out['7']['boxes'] and out['7']['gt_boxes'] == [[1, 1, 2, 2]]
