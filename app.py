import glob
from seg_track_anything import draw_mask
from aot_tracker import _palette

import gradio as gr
import platform
from gui.video_obj import Video_obj
from gui.tracker import Tracker

os_sys = platform.uname().system




def show_page():
    return gr.update(visible=True)





def clean_fn():
    return None, None




def init_video(obj, input, progress=gr.Progress()):
    obj = Video_obj(input, progress)
    gr.Info("video load ....ok")
    return obj, gr.update(maximum=obj.n_frame-1), gr.update(maximum=obj.n_frame-1), gr.update(maximum=obj.n_frame-1, value=obj.n_frame-1), gr.update(visible=True), obj.display(0, "Image")

def add_object_fn(obj, name):
    if len(name) == 0:
        return obj, "", f"Empty Object name"

    n = obj.add_object(name)
    return obj, "", f"Add Object, number of Object: {n}"



def edit_tab_fn(v, tk, start, stop, device):
    txt = """
    Select Start & Stop Frame. 
    Label on start frame and track until stop frame."
    """
    gr.Info(txt)
    ff = frame_index_slide_fn(v, start, "Image")  # ff: First Frame
    tk = Tracker(ff, start, stop, device)
    return ff, tk, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), True



def view_tab_fn():

    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), False


def frame_index_slide_fn(obj, index, mode):
    out = obj.display(int(index), mode)

    return out


def select_start_frame_slide_fn(tk_cfg, start):  # [l, r]
    tk_cfg["begin"] = start
    return tk_cfg

def long_term_mem_gap_fn(tk_cfg, gap):
    tk_cfg["gap"] = int(gap)
    return tk_cfg

def max_len_long_term_fn(tk_cfg, max_len):
    tk_cfg["max_len"] = int(max_len)
    return tk_cfg

def select_stop_frame_slide_fn(tk_cfg, stop):
    tk_cfg["end"] = int(stop)
    return tk_cfg



def apply_btm_click_fn(v, tk, tk_cfg):
    print(tk_cfg)
    ff = frame_index_slide_fn(v, tk_cfg["begin"], "Image")  # ff: First Frame
    tk = Tracker(ff, tk_cfg["begin"], tk_cfg["end"],tk_cfg["device"],tk_cfg["gap"], tk_cfg["max_len"])
    return ff, tk

def track_click_fn(tk, click_type_drop, ok, evt: gr.SelectData):
    if ok != True:
        gr.Info("Not Edit Mode!")
        return tk, None

    mode = 1 if click_type_drop == "Positive" else 0
    point = (evt.index[0], evt.index[1])  # (y, x)

    _, mix = tk.click(point, mode)

    gr.Info("click!")
    return tk, mix


def track_click_next_fn(tk):
    tk.next_obj()
    gr.Info("Label next object.")
    return tk


def tracking_btm_fn(tk, v, stop, progress=gr.Progress()):
    gr.Info(f"Tracking start, from {tk.start} to {tk.stop}.")
    tk.stop = stop
    tk.tracking(v, progress)
    gr.Info(f"Tracking Done, from {tk.start} to {tk.stop}.")
    tk.segtracker.sam.sam.to('cpu')
    del tk
    return None, gr.update(visible=True), "Done"


def undo_btm_fn(tk):
    _, mix = tk.undo()
    gr.Info("Relabel this object")

    return tk, mix



def generate_output_btm_fn(v):
    n_mask = len(glob.glob(f'{v.mask_dir}*.png'))
    n_mix = len(glob.glob(f'{v.mix_dir}*.png'))

    print(n_mask, n_mix)
    if n_mask != v.n_frame or n_mix != v.n_frame:
        gr.Info("Error")
        return None, None, None

    mask_video = v.write_video('mask')
    mix_video = v.write_video('mix')
    gr.Info("OK")
    return mask_video, mix_video, None


app = gr.Blocks()

with app:
    # Variable
    video_obj = gr.State(None)
    tracker_obj = gr.State(None)
    is_click = gr.State(None)
    tk_cfg = gr.State({"begin": 0, "end": 10, "max_len": 50, "gap": 30, "device": "mps" if os_sys == 'Darwin' else "cuda"})

    # Front-end

    reset_btm = gr.Button("Reset", interactive=True, visible=False)
    with gr.Row(visible=True) as file_page:
            input_video = gr.File(label='Input video')

                
                
            

    with gr.Row(visible=True) as edit_page:
        with gr.Column(scale=0.8):
            display_img = gr.Image(label='Display', interactive=False)
            frame_index_slide = gr.Slider(label="Frame Index",
                                          minimum=0, step=1, maximum=10,
                                          value=0, interactive=True)

            

        with gr.Column(scale=0.2):
            # display_txt = gr.Textbox(label="Log")
            with gr.Tab(label="View") as view_tab:
                display_mode_drop = gr.Dropdown(
                    choices=["Image", "Image & Mask", "Mask"],
                    label="Display Mode", value="Image",
                    interactive=True)

                step_frame_num_text = gr.Dropdown(
                    choices=["1", "5", "10", "100", "1000", "10000"],
                    label="Frame step number", value="1",
                    interactive=True)

            with gr.Tab(label="Edit") as edit_tab:
                device_drop = gr.Dropdown(
                    choices=["cuda", "cpu", "mps"],
                    label="Device", value="mps" if os_sys == 'Darwin' else "cuda",
                    interactive=True)
                long_term_mem_gap =  gr.Number(label="long_term_mem_gap", value=30, interactive=True)
                max_len_long_term =  gr.Number(label="max_len_long_term", value=50, interactive=True)
                select_start_frame_slide = gr.Slider(
                label="Enter Start Frame Index",
                    minimum=0, step=1, maximum=10, value=0, interactive=True, visible=False)

                select_stop_frame_slide = gr.Slider(
                label="Enter Stop Frame Index",
                    minimum=0, step=1, maximum=10, value=10, interactive=True, visible=False)

                apply_btm =  gr.Button("Apply Setting", interactive=True)
                click_type_drop = gr.Dropdown(
                    choices=["Positive", "Negative"],
                    label="Click Type", value="Positive",
                    interactive=True)
                undo_btm = gr.Button("Undo", interactive=True, visible=False)
                click_done_btm = gr.Button("Next Object", interactive=True)

                tracking_btm = gr.Button("Tracking", interactive=True)
                progress_bar = gr.Textbox(label="Progress")
                output_page_btm = gr.Button(
                    "Open Output Setting Page", interactive=True, visible=False)

    with gr.Row(visible=False) as output_page:
        generate_output_btm = gr.Button("Output Video", interactive=True)
        mask_video = gr.File(label='Mask Video', interactive=False)
        mix_video = gr.File(label='Mask & Frame Video', interactive=False)
        object_config = gr.File(label='Object Config',
                                interactive=False, visible=False)

    

    # Function

    input_video.change(
        fn=init_video,
        inputs=[video_obj, input_video],
        outputs=[video_obj, frame_index_slide, select_start_frame_slide,
                 select_stop_frame_slide, edit_page, display_img],
    )

    edit_tab.select(
        fn=edit_tab_fn,
        inputs=[video_obj, tracker_obj,
                select_start_frame_slide, select_stop_frame_slide, device_drop],
        outputs=[display_img, tracker_obj, frame_index_slide, select_start_frame_slide,
                 select_stop_frame_slide, is_click],

    )

    view_tab.select(
        fn=view_tab_fn,
        outputs=[frame_index_slide, select_start_frame_slide,
                 select_stop_frame_slide, is_click],

    )

    frame_index_slide.change(
        fn=frame_index_slide_fn,
        inputs=[video_obj, frame_index_slide, display_mode_drop],
        outputs=[display_img],
    )

    select_start_frame_slide.release(
        fn=select_start_frame_slide_fn,
        inputs=[tk_cfg, select_start_frame_slide],
        outputs=[tk_cfg],
    )

    select_stop_frame_slide.release(
        fn=select_stop_frame_slide_fn,
        inputs=[tk_cfg, select_stop_frame_slide],
        outputs=[tk_cfg],
    )

    long_term_mem_gap.blur(
        fn=long_term_mem_gap_fn,
        inputs=[tk_cfg, long_term_mem_gap],
        outputs=[tk_cfg],
    )
    max_len_long_term.blur(
        fn=max_len_long_term_fn,
        inputs=[tk_cfg, max_len_long_term],
        outputs=[tk_cfg],
    )

    display_img.select(
        fn=track_click_fn,
        inputs=[tracker_obj, click_type_drop, is_click],
        outputs=[tracker_obj, display_img],
    )

    click_done_btm.click(
        fn=track_click_next_fn,
        inputs=[tracker_obj],
        outputs=[tracker_obj],
    )
    apply_btm.click(
        fn=apply_btm_click_fn,
        inputs=[video_obj, tracker_obj,tk_cfg],
        outputs=[display_img, tracker_obj],
    )

    undo_btm.click(
        fn=undo_btm_fn,
        inputs=[tracker_obj],
        outputs=[tracker_obj, display_img],
    )

    tracking_btm.click(
        fn=tracking_btm_fn,
        inputs=[tracker_obj, video_obj, select_stop_frame_slide],
        outputs=[tracker_obj, output_page, progress_bar],
    )

    output_page_btm.click(
        fn=show_page,
        outputs=[output_page]
    )

    generate_output_btm.click(
        fn=generate_output_btm_fn,
        inputs=[video_obj],
        outputs=[mask_video, mix_video, object_config]
    )

    reset_btm.click(
        fn=clean_fn,
        outputs=[video_obj, tracker_obj]
    )

if __name__ == "__main__":
    app.queue(concurrency_count=5)
    app.launch(server_name="0.0.0.0").queue()
