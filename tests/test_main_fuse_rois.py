import json

import main_fuse_rois


def test_fuse_frame_unions_boxes_drops_duplicates_and_caps():
    a = {'rois': [[0, 0, 20, 20, 0.9], [40, 40, 60, 60, 0.3]], 'boxes': [[5, 5, 15, 15, 0.9], [45, 45, 55, 55, 0.3]], 'gt_boxes': [[6, 6, 14, 14]]}
    b = {'rois': [[1, 1, 21, 21, 0.8], [70, 70, 90, 90, 0.5]], 'boxes': [[5, 5, 15, 16, 0.8], [75, 75, 85, 85, 0.5]], 'gt_boxes': [[6, 6, 14, 14]]}
    f = main_fuse_rois.fuse_frame([a, b], iou_dedup=0.7)
    # the near-duplicate of the 0.9 box (IoU 0.91) is dropped, the rest is the union sorted by confidence
    assert [r[4] for r in f['rois']] == [0.9, 0.5, 0.3] and f['boxes'][1][:4] == [75, 75, 85, 85] and f['gt_boxes'] == [[6, 6, 14, 14]]
    assert main_fuse_rois.fuse_frame([a, b], iou_dedup=1.0)['rois'][1][4] == 0.8              # 1.0 keeps everything
    assert [r[4] for r in main_fuse_rois.fuse_frame([a, b], iou_dedup=0.7, max_rois=2)['rois']] == [0.9, 0.5]
    assert main_fuse_rois.fuse_frame([{'rois': [], 'boxes': [], 'gt_boxes': [[1, 1, 2, 2]]}], 0.7) == {'rois': [], 'boxes': [], 'gt_boxes': [[1, 1, 2, 2]]}


def test_main_writes_the_fused_export_over_the_union_of_frames(tmp_path, capsys):
    e1 = {'7': {'rois': [[0, 0, 20, 20, 0.9]], 'boxes': [[5, 5, 15, 15, 0.9]], 'gt_boxes': [[6, 6, 14, 14]]}, '8': {'rois': [], 'boxes': [], 'gt_boxes': [[30, 30, 40, 40]]}}
    e2 = {'8': {'rois': [[28, 28, 42, 42, 0.4]], 'boxes': [[31, 31, 39, 39, 0.4]], 'gt_boxes': [[30, 30, 40, 40]]}, '9': {'rois': [], 'boxes': [], 'gt_boxes': []}}
    (tmp_path / 'a.json').write_text(json.dumps(e1)); (tmp_path / 'b.json').write_text(json.dumps(e2))
    main_fuse_rois.main(main_fuse_rois.get_args_parser().parse_args(['--inputs', str(tmp_path / 'a.json'), str(tmp_path / 'b.json'), '--name', 'fz', '--out_dir', str(tmp_path)]))
    out = json.loads((tmp_path / 'rois_fz_test.json').read_text())
    assert set(out) == {'7', '8', '9'} and len(out['8']['rois']) == 1 and out['9']['rois'] == []
    msg = capsys.readouterr().out
    assert '2 exports, 2 boxes in -> 2 out over 3 frames, 1 frames without any box, 2/2 GT boxes fully inside an ROI (1.000)' in msg
