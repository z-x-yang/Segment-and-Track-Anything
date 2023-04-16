import gradio as gr
import importlib
import sys
import os

# sys.path.append('.')
# sys.path.append('..')

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
from seg_track_anything import seg_track_anything

def predict(input_video_file, model, sam_gap, max_obj_num, points_per_side):

    return seg_track_anything(input_video_file, model, sam_gap, max_obj_num, points_per_side)


def video_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_video_file = gr.Video(label='input_video')

                with gr.Row():
                    with gr.Column():
                        aot_model = gr.Dropdown(
                            label="aot_model",
                            choices = [
                                "deaotb",
                                "deaotl",
                                "r50_deaotl"
                            ],
                            value = "r50_deaotl",
                        )
                        sam_gap = gr.Slider(
                            label='sam_gap',
                            minimum = 1,
                            step=1,
                            maximum = 9999,
                            value=4
                        )
                        points_per_side = gr.Slider(
                            label = "points_per_side",
                            minimum= 1,
                            step = 1,
                            maximum=30,
                            value=16
                        )
                        max_obj_num = gr.Slider(
                            label='max_obj_num',
                            minimum = 50,
                            step=1,
                            maximum = 300,
                            value=255
                        )
                seg_automask_video_predict = gr.Button(value="Seg and Track")

            with gr.Column():
                output_video = gr.Video(label='output')
                output_mask = gr.File(label="predicted_mask")

        seg_automask_video_predict.click(
            fn=predict,
            inputs=[
                input_video_file,
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side
            ],
            outputs=[
                output_video,
                output_mask
            ],
        )
    # gr.Examples(
    # examples=[
    #     os.path.join(os.path.dirname(__file__), "assets", "840_iSXIa0hE8Ek.mp4"),
    #     os.path.join(os.path.dirname(__file__), "assets", "bear.mp4"),
    #     os.path.join(os.path.dirname(__file__), "assets", "camel.mp4"),
    #     os.path.join(os.path.dirname(__file__), "assets", "skate-park.mp4"),
    #     os.path.join(os.path.dirname(__file__), "assets", "swing.mp4"),
    #     ],
    # inputs=[input_video_file],
    # outputs=[
    #     output_video,
    #     output_mask
    #     ],
    # fn=predict,
    # # cache_examples=True,
    # )


def metaseg_app():
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

        with gr.Row():
            with gr.Column():
                with gr.Tab("Video"):
                    video_app()
        
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)


if __name__ == "__main__":
    metaseg_app()