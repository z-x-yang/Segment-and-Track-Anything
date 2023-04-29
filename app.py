from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os

from matplotlib.pyplot import step

from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker

# sys.path.append('.')
# sys.path.append('..')

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
import gc
import numpy as np
import json
from tool.transfer_tools import mask2bbox

def clean():
    return None, None, None, None, None, None, [[], []]

def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )
    
    prompt = {
        "points_coord":click_stack[0],
        "points_mode":click_stack[1],
        "multimask":"True",
    }

    return prompt

def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""

    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    
    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame, ""

def get_meta_from_img_seq(input_img_seq):
    if input_img_seq is None:
        return None, None, None, ""

    print("get meta information of img seq")
    # Create dir
    file_name = input_img_seq.name.split('/')[-1].split('.')[0]
    file_path = f'./assets/{file_name}'
    if os.path.isdir(file_path):
        os.system(f'rm -r {file_path}')
    os.makedirs(file_path)
    # Unzip file
    os.system(f'unzip {input_img_seq.name} -d ./assets ')
    
    imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
    first_frame = imgs_path[0]
    first_frame = cv2.imread(first_frame)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker

def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    
    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""

def init_SegTracker_Stroke(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    
    if origin_frame is None:
        return None, origin_frame, [[], []], origin_frame

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term

    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], origin_frame

def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    
    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]

    print("Undo!")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]
    
    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask":"True",
        }

        masked_frame = seg_acc_click(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_stack
    else:
        return Seg_Tracker, origin_frame, [[], []]


def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                      origin_frame=origin_frame, 
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame

def sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, evt:gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """

    print("Click")

    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        # TODO：add everything positive points
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    # get click prompts for sam to predict mask
    click_prompt = get_click_prompt(click_stack, point)

    # Refine acc to prompt
    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_stack

def sam_stroke(Seg_Tracker, origin_frame, drawing_board, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):

    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)
    
    print("Stroke")
    mask = drawing_board["mask"]
    bbox = mask2bbox(mask[:, :, 0])  # bbox: [[x0, y0], [x1, y1]]
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_bbox(origin_frame, bbox)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame

def gd_detect(Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)


    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame

def segment_everything(Seg_Tracker, aot_model, long_term_mem, max_len_long_term, origin_frame, sam_gap, max_obj_num, points_per_side):
    
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Everything")

    frame_idx = 0

    with torch.cuda.amp.autocast():
        pred_mask = Seg_Tracker.seg(origin_frame)
        torch.cuda.empty_cache()
        gc.collect()
        Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)
        Seg_Tracker.first_frame_mask = pred_mask

    masked_frame = draw_mask(origin_frame.copy(), pred_mask)

    return Seg_Tracker, masked_frame

def add_new_object(Seg_Tracker):

    prev_mask = Seg_Tracker.first_frame_mask
    Seg_Tracker.update_origin_merged_mask(prev_mask)
    print("Ready to add new object!")

    return Seg_Tracker, [[], []]

def tracking_objects(Seg_Tracker, input_video, input_img_seq, fps):
    print("Start tracking !")
    return tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps)

def seg_track_app():

    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Segment and Track Anything(SAM-Track)</span>
            </div>
            '''
        )

        click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

        with gr.Row():
            # video input
            with gr.Column(scale=0.5):

                tab_video_input = gr.Tab(label="Video type input")
                with tab_video_input:
                    input_video = gr.Video(label='Input video').style(height=550)
                
                tab_img_seq_input = gr.Tab(label="Image-Seq type input")
                with tab_img_seq_input:
                    with gr.Row():
                        input_img_seq = gr.File(label='Input Image-Seq').style(height=550)
                        with gr.Column(scale=0.25):
                            extract_button = gr.Button(value="extract")
                            fps = gr.Slider(label='fps', minimum=5, maximum=50, value=8, step=1)

                input_first_frame = gr.Image(label='Segment result of first frame',interactive=True).style(height=550)


                tab_everything = gr.Tab(label="Everything")
                with tab_everything:
                    with gr.Row():
                        seg_every_first_frame = gr.Button(value="Segment everything for first frame", interactive=True)
                        point_mode = gr.Radio(
                            choices=["Positive"],
                            value="Positive",
                            label="Point Prompt",
                            interactive=True)

                        every_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )

                            # every_reset_but = gr.Button(
                            #             value="Reset",
                            #             interactive=True
                            #                     )

                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        point_mode = gr.Radio(
                                    choices=["Positive",  "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True)

                        # args for modify and tracking 
                        click_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )
                            # click_reset_but = gr.Button(
                            #             value="Reset",
                            #             interactive=True
                            #                     )

                tab_stroke = gr.Tab(label="Stroke")
                with tab_stroke:
                    drawing_board = gr.Image(label='Drawing Board', tool="sketch", brush_radius=10, interactive=True)
                    with gr.Row():
                        seg_acc_stroke = gr.Button(value="Segment", interactive=True)
                        # stroke_reset_but = gr.Button(
                        #                 value="Reset",
                        #                 interactive=True
                        #                         )
                
                tab_text = gr.Tab(label="Text")
                with tab_text:
                    grounding_caption = gr.Textbox(label="Detection Prompt")
                    detect_button = gr.Button(value="Detect")
                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            with gr.Column(scale=0.5):
                                box_threshold = gr.Slider(
                                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )
                            with gr.Column(scale=0.5):
                                text_threshold = gr.Slider(
                                    label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )

                with gr.Row():
                    with gr.Column(scale=0.5):
                        with gr.Tab(label="SegTracker Args"):
                            # args for tracking in video do segment-everthing
                            points_per_side = gr.Slider(
                                label = "points_per_side",
                                minimum= 1,
                                step = 1,
                                maximum=100,
                                value=16,
                                interactive=True
                            )

                            sam_gap = gr.Slider(
                                label='sam_gap',
                                minimum = 1,
                                step=1,
                                maximum = 9999,
                                value=100,
                                interactive=True,
                            )

                            max_obj_num = gr.Slider(
                                label='max_obj_num',
                                minimum = 50,
                                step=1,
                                maximum = 300,
                                value=255,
                                interactive=True
                            )
                            with gr.Accordion("aot advanced options", open=False):
                                aot_model = gr.Dropdown(
                                    label="aot_model",
                                    choices = [
                                        "deaotb",
                                        "deaotl",
                                        "r50_deaotl"
                                    ],
                                    value = "r50_deaotl",
                                    interactive=True,
                                )
                                long_term_mem = gr.Slider(label="long term memory gap", minimum=1, maximum=9999, value=9999, step=1)
                                max_len_long_term = gr.Slider(label="max len of long term memory", minimum=1, maximum=9999, value=9999, step=1)

                    with gr.Column():
                        new_object_button = gr.Button(
                            value="Add new object", 
                            interactive=True
                        )
                        reset_button = gr.Button(
                            value="Reset",
                            interactive=True,
                        )
                        track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                ).style(size="lg")

            with gr.Column(scale=0.5):
                output_video = gr.Video(label='Output video').style(height=550)
                output_mask = gr.File(label="Predicted masks")

    ##########################################################
    ######################  back-end #########################
    ##########################################################

        # listen to the input_video to get the first frame of video
        input_video.change(
            fn=get_meta_from_video,
            inputs=[
                input_video
            ],
            outputs=[
                input_first_frame, origin_frame, drawing_board, grounding_caption
            ]
        )

        # listen to the input_img_seq to get the first frame of video
        input_img_seq.change(
            fn=get_meta_from_img_seq,
            inputs=[
                input_img_seq
            ],
            outputs=[
                input_first_frame, origin_frame, drawing_board, grounding_caption
            ]
        )
        
        #-------------- Input compont -------------
        tab_video_input.select(
            fn = clean,
            inputs=[],
            outputs=[
                input_video,
                input_img_seq,
                Seg_Tracker,
                input_first_frame,
                origin_frame,
                drawing_board,
                click_stack,
            ]
        )

        tab_img_seq_input.select(
            fn = clean,
            inputs=[],
            outputs=[
                input_video,
                input_img_seq,
                Seg_Tracker,
                input_first_frame,
                origin_frame,
                drawing_board,
                click_stack,
            ]
        )

        extract_button.click(
            fn=get_meta_from_img_seq,
            inputs=[
                input_img_seq
            ],
            outputs=[
                input_first_frame, origin_frame, drawing_board
            ]
        )


        # ------------------- Interactive component -----------------

        # listen to the tab to init SegTracker
        tab_everything.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term, 
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption
            ],
            queue=False,
            
        )
        
        tab_click.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption
            ],
            queue=False,
        )

        tab_stroke.select(
            fn=init_SegTracker_Stroke,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, drawing_board
            ],
            queue=False,
        )

        tab_text.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption
            ],
            queue=False,
        )

        # Use SAM to segment everything for the first frame of video
        seg_every_first_frame.click(
            fn=segment_everything,
            inputs=[
                Seg_Tracker,
                aot_model,
                long_term_mem,
                max_len_long_term,
                origin_frame,
                sam_gap,
                max_obj_num,
                points_per_side,

            ],
            outputs=[
                Seg_Tracker,
                input_first_frame,
            ],
            )
        
        # Interactively modify the mask acc click
        input_first_frame.select(
            fn=sam_click,
            inputs=[
                Seg_Tracker, origin_frame, point_mode, click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )

        # Interactively segment acc stroke
        seg_acc_stroke.click(
            fn=sam_stroke,
            inputs=[
                Seg_Tracker, origin_frame, drawing_board,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, drawing_board
            ]
        )

        # Use grounding-dino to detect object
        detect_button.click(
            fn=gd_detect, 
            inputs=[
                Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold,
                aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side
                ], 
            outputs=[
                Seg_Tracker, input_first_frame
                ]
                )

        # Add new object
        new_object_button.click(
            fn=add_new_object,
            inputs=
            [
                Seg_Tracker
            ],
            outputs=
            [
                Seg_Tracker, click_stack
            ]
        )

        # Track object in video
        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
                input_img_seq,
                fps,
            ],
            outputs=[
                output_video, output_mask
            ]
        )

        # ----------------- Reset and Undo ---------------------------

        # Rest 
        reset_button.click(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption
            ],
            queue=False,
            show_progress=False
        ) 

        # every_reset_but.click(
        #     fn=init_SegTracker,
        #     inputs=[
        #         aot_model,
        #         sam_gap,
        #         max_obj_num,
        #         points_per_side,
        #         origin_frame
        #     ],
        #     outputs=[
        #         Seg_Tracker, input_first_frame, click_stack, grounding_caption
        #     ],
        #     queue=False,
        #     show_progress=False
        # ) 

        # click_reset_but.click(
        #     fn=init_SegTracker,
        #     inputs=[
        #         aot_model,
        #         sam_gap,
        #         max_obj_num,
        #         points_per_side,
        #         origin_frame
        #     ],
        #     outputs=[
        #         Seg_Tracker, input_first_frame, click_stack, grounding_caption
        #     ],
        #     queue=False,
        #     show_progress=False
        # ) 

        # stroke_reset_but.click(
        #     fn=init_SegTracker_Stroke,
        #     inputs=[
        #         aot_model,
        #         sam_gap,
        #         max_obj_num,
        #         points_per_side,
        #         origin_frame,
        #     ],
        #     outputs=[
        #         Seg_Tracker, input_first_frame, click_stack, drawing_board
        #     ],
        #     queue=False,
        #     show_progress=False
        # )

        # Undo click
        click_undo_but.click(
            fn = undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
               Seg_Tracker, input_first_frame, click_stack
            ]
        )

        every_undo_but.click(
            fn = undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
               Seg_Tracker, input_first_frame, click_stack
            ]
        )
        
        with gr.Tab(label='Video example'):
            gr.Examples(
                examples=[
                    # os.path.join(os.path.dirname(__file__), "assets", "840_iSXIa0hE8Ek.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "blackswan.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "Resized_cxk.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "bear.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "camel.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "skate-park.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "swing.mp4"),
                    ],
                inputs=[input_video],
            )
        
        with gr.Tab(label='Image-seq expamle'):
            gr.Examples(
                examples=[
                    os.path.join(os.path.dirname(__file__), "assets", "840_iSXIa0hE8Ek.zip"),
                ],
                inputs=[input_img_seq],
            )
    
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)


if __name__ == "__main__":
    seg_track_app()