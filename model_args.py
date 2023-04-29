# Explanation of generator_args is in sam/segment_anything/automatic_mask_generator.py: SamAutomaticMaskGenerator
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
    'max_len_long_term': 9999,
    'gpu_id': 0,
}
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80% 
}