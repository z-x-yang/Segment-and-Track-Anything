sam_args = {
    'sam_checkpoint': "ckpt/sam_vit_b_01ec64.pth",
    'model_type': "vit_b",
    'generator_args':{
        'points_per_side': 16,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    },
    'gpu_id': 0,
}
aot_args = {
    'phase': 'PRE_YTB_DAV',
    'model': 'r50_deaotl',
    'model_path': 'ckpt/R50_DeAOTL_PRE_YTB_DAV.pth',
    'long_term_mem_gap': 9999,
    'gpu_id': 0,
}
segtracker_args = {
    'sam_gap': 4,
    'match_iou_thr': 0.5,
    'min_area': 200,
    'max_obj_num': 255,
    'min_new_obj_iou': 0.8,
}