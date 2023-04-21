from PIL.ImageOps import scale
from blib2to3.pgen2.grammar import Label
import gradio as gr
import importlib
import sys
import os
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

def pause_video(play_state):
    print("user pause_video")
    play_state.append(time.time())
    return play_state

def play_video(play_state):
    print("user play_video")
    play_state.append(time.time())
    return play_state

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

def get_meta_from_video(input_video):
    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame,cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame

def init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame):
    
    if origin_frame is None:
        return None, origin_frame, [[], []]

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]

    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []]

def init_SegTracker_Brush(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame):
    
    if origin_frame is None:
        return None, origin_frame, [[], []], origin_frame

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]

    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], origin_frame

def undo_click_state_and_refine_seg(Seg_Tracker, origin_frame, click_state, aot_model, sam_gap, max_obj_num, points_per_side):

    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]
    
    if len(click_state[0]) > 0:
        click_state[0] = click_state[0][: -1]
        click_state[1] = click_state[1][: -1]
    
    if len(click_state[0]) > 0:
        prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
        }

        masked_frame = refine_acc_prompt(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_state
    else:
        # Seg_Tracker, _, _ = init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)
        return Seg_Tracker, origin_frame, [[], []]


def refine_acc_prompt(Seg_Tracker, prompt, origin_frame):
    # Refine acc to prompt
    predicted_mask, masked_frame = Seg_Tracker.refine_first_frame_click( 
                                                      origin_frame=origin_frame, 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                    )

    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)

    return masked_frame

def sam_refine(Seg_Tracker, origin_frame, point_prompt, click_state, aot_model, sam_gap, max_obj_num, points_per_side, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        # TODO：add everything positive points
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
    
    # default value
    # points = np.array([[evt.index[0],evt.index[1]]])
    # labels= np.array([1])
    # if len(logit)==0:
    #     logit = None

    if Seg_Tracker is None:
        Seg_Tracker, _, _ = init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)

    # prompt for sam model
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    # Refine acc to prompt
    masked_frame =  refine_acc_prompt(Seg_Tracker, prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_state

def sam_brush(Seg_Tracker, origin_frame, brush_plane, aot_model, sam_gap, max_obj_num, points_per_side):

    if Seg_Tracker is None:
        Seg_Tracker, _ , _ = init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)

    mask = brush_plane["mask"]
    bbox = mask2bbox(mask[:, :, 0])  # bbox: [[x0, y0], [x1, y1]]
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_bbox(origin_frame, bbox)

    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)

    return Seg_Tracker, masked_frame, origin_frame

def segment_everything(Seg_Tracker, aot_model, origin_frame, sam_gap, max_obj_num, points_per_side):
    
    if Seg_Tracker is None:
        Seg_Tracker, _ , _ = init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)

    frame_idx = 0

    with torch.cuda.amp.autocast():
        pred_mask = Seg_Tracker.seg(origin_frame)
        torch.cuda.empty_cache()
        gc.collect()
        Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)

        masked_frame = draw_mask(origin_frame.copy(), pred_mask)
        # masked_frame = (origin_frame*0.3+colorize_mask(pred_mask)*0.7).astype(np.uint8)

    return Seg_Tracker, masked_frame

def tracking_objects(Seg_Tracker, input_video):
    return tracking_objects_in_video(Seg_Tracker, input_video)

def seg_track_app():

    ##########################################################
    ######################  Front-end ########################
    ##########################################################

    app = gr.Blocks()
    with app:
        gr.Markdown("# **<h1 align='center'>Segment and Track Anything for Video</h1>**")
        gr.Markdown(
            """
            <h3 center>
            The algorithm for segment and track anything (SegTracker) in a video. 
            The current implemented pipeline consists of a segmentation method <a href='https://github.com/facebookresearch/segment-anything' target='_blank'>SAM (Segment Anything Model)</a>
            and a VOS method <a href='https://github.com/yoxu515/aot-benchmark' target='_blank'>AOT (Associating Object with Transformers)</a> . The pipeline automaticly and dynamically detects and segments new objects by SAM, and tracks all spotted objects by AOT.        
            </center>
            """
        )  
        
        """
        state for 
        """
        state = gr.State([])
        play_state = gr.State([])
        video_state = gr.State([[],[],[]])
        click_state = gr.State([[],[]])
        logits = gr.State([])
        masks = gr.State([])
        origin_frame = gr.State(None)
        select_correction_frame = gr.State(None)
        corrected_state = gr.State([[],[],[]])
        tab_state = gr.State("everything")
        Seg_Tracker = gr.State(None)

        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

        with gr.Row():
            # video input
            with gr.Column(scale=0.5):
                input_video = gr.Video(label='input_video').style(height=380)
                # listen to the user action for play and pause input video
                input_video.play(fn=play_video, inputs=play_state, outputs=play_state, scroll_to_output=True, show_progress=True)
                input_video.pause(fn=pause_video, inputs=play_state, outputs=play_state)

                input_video_first_frame = gr.Image(label='input_video_first_frame', interactive=True).style(height=380)


                tab_everything = gr.Tab(label="Segment everything")
                with tab_everything:
                    with gr.Row():
                        seg_every_first_frame = gr.Button(value="Segment everything for first frame", interactive=True)
                        point_prompt = gr.Radio(
                            choices=["Positive"],
                            value="Positive",
                            label="Point Prompt",
                            interactive=True)

                        every_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )

                        every_reset_but = gr.Button(
                                    value="Reset",
                                    interactive=True
                                            )

                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        point_prompt = gr.Radio(
                                    choices=["Positive",  "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True)
                        # args for modify and tracking 
                        click_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )

                        click_reset_but = gr.Button(
                                    value="Reset",
                                    interactive=True
                                            )

                tab_brush = gr.Tab(label="Brush")
                with tab_brush:
                    bursh_plane = gr.Image(label='bursh_plane', tool="sketch", interactive=True).style(height=380)
                    seg_acc_brush = gr.Button(value="Segment", interactive=True)

                with gr.Row():
            
                    # args for tracking in video do segment-everthing
                    with gr.Column(scale=0.5):

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

                        points_per_side = gr.Slider(
                            label = "points_per_side",
                            minimum= 1,
                            step = 1,
                            maximum=100,
                            value=16,
                            interactive=True
                        )


                    with gr.Column(scale=0.5):
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
                track_for_video = gr.Button(
                    value="Start Tracking",
                    interactive=True
                    )

            with gr.Column(scale=0.5):
                output_video = gr.Video(label='output').style(height=380)

                # TODO: V2-Interactively correct intermediate frames
                # image_output = gr.Image(type="pil", interactive=True, elem_id="image_output").style(height=360)
                # image_selection_slider = gr.Slider(minimum=0, maximum=100, step=0.1, value=0, label="Image Selection", interactive=True)
                # correct_track_button = gr.Button(value="Interactive Correction")

                output_mask = gr.File(label="predicted_mask")

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
                input_video_first_frame, origin_frame, bursh_plane
            ]
        )

        # listen to the tab to init SegTracker
        tab_everything.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, click_state
            ],
            queue=False,
            
        )
        
        tab_click.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, click_state
            ],
            queue=False,
        )

        tab_brush.select(
            fn=init_SegTracker_Brush,
            inputs=[
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, click_state, bursh_plane
            ],
            queue=False,
        )

        # Init Seg-Tracker
        # initial_seg_tracker.click(
        #     fn=init_SegTracker,
        #     inputs=[
        #         aot_model,
        #         sam_gap,
        #         max_obj_num,
        #         points_per_side,
        #         origin_frame
        #     ],
        #     outputs=[
        #         Seg_Tracker, input_video_first_frame, click_state
        #     ]
        # )

        # Use SAM to segment everything for the first frame of video
        seg_every_first_frame.click(
            fn=segment_everything,
            inputs=[
                Seg_Tracker,
                aot_model,
                origin_frame,
                sam_gap,
                max_obj_num,
                points_per_side,

            ],
            outputs=[
                Seg_Tracker,
                input_video_first_frame,
            ],
            )
        
        # Interactively modify the mask acc click
        input_video_first_frame.select(
            fn=sam_refine,
            inputs=[
                Seg_Tracker, origin_frame, point_prompt, click_state,
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, click_state
            ]
        )

        # Interactively segment acc brush
        seg_acc_brush.click(
            fn=sam_brush,
            inputs=[
                Seg_Tracker, origin_frame, bursh_plane,
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, bursh_plane
            ]
        )



        # Track object in video
        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
            ],
            outputs=[
                output_video, output_mask
            ]
        )

        # Reset SegTracker and click_state
        click_reset_but.click(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, click_state
            ],
            queue=False,
            show_progress=False
        ) 

        every_reset_but.click(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_video_first_frame, click_state
            ],
            queue=False,
            show_progress=False
        ) 

        # Undo click
        click_undo_but.click(
            fn = undo_click_state_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_state,
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
               Seg_Tracker, input_video_first_frame, click_state
            ]
        )

        every_undo_but.click(
            fn = undo_click_state_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_state,
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
               Seg_Tracker, input_video_first_frame, click_state
            ]
        )

        gr.Examples(
            examples=[
                os.path.join(os.path.dirname(__file__), "assets", "840_iSXIa0hE8Ek.mp4"),
                os.path.join(os.path.dirname(__file__), "assets", "blackswan.mp4"),
                os.path.join(os.path.dirname(__file__), "assets", "Resized_cxk.mp4"),

                # os.path.join(os.path.dirname(__file__), "assets", "bear.mp4"),
                # os.path.join(os.path.dirname(__file__), "assets", "camel.mp4"),
                # os.path.join(os.path.dirname(__file__), "assets", "skate-park.mp4"),
                # os.path.join(os.path.dirname(__file__), "assets", "swing.mp4"),
                ],
            inputs=[input_video],
        )


    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)


if __name__ == "__main__":
    seg_track_app()