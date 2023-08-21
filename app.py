import glob
from matplotlib import pyplot as plt
from SegTracker import SegTracker
from seg_track_anything import draw_mask
from aot_tracker import _palette
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import datetime
import gradio as gr
import os
DEVICE = 'mps'
HEADER = """
<div style="text-align:center;">
    <span style="font-size:3em; font-weight:bold;">Tracking Gold Apple</span>
</div>
"""


def swap_page():
    return gr.update(visible=False), gr.update(visible=True)


def show_page():
    return gr.update(visible=True)

def build_putpalette(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    return save_mask

class Video_obj:
    def __init__(self, file, progress) -> None:
        self.video = file.name
        self.build_folder()
        self.read_video(progress)



    def build_folder(self) -> None:
        now = datetime.datetime.today()
        now = str(now).split('.')[0].replace(':', '-').replace(' ', '-')
        self.folder = f'result/{now}/'
        self.frame_dir = f'{self.folder}frame/'
        self.mask_dir = f'{self.folder}mask/'
        self.mix_dir = f'{self.folder}mix/'
        self.obj_li = []
        for it in [self.folder, self.frame_dir, self.mask_dir, self.mix_dir]:
            os.makedirs(it, exist_ok=True)



    def read_video(self, progress):
        cap = cv2.VideoCapture(self.video)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in progress.tqdm(range(self.n_frame)):
            ok, frame = cap.read()
            assert ok, "[E] Read Frame Error"

            name = f'{self.frame_dir}{i:06d}.png'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(name, frame)

        cap.release()

    def write_video(self, mode):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        name = f'{self.folder}mode.mp4'
        video = cv2.VideoWriter(name, fourcc, self.fps, (self.width, self.height))
        if mode == 'mask':
            li = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        else:
            li = sorted(glob.glob(os.path.join(self.mix_dir, '*.png')))
        
        for img_file in li:
            img = cv2.imread(img_file)
            video.write(img)

        video.release()

        return name


    def add_object(self, name):
        self.obj_li.append(name)
        return len(self.obj_li)
    
    def display(self, i, mode):
        if i < 0 or i >= self.n_frame:
            return None

        if mode == 'Image':
            try:
                return np.array(Image.open(f'{self.frame_dir}{i:06d}.png'))
            except:
                return None
        
        if mode == 'Image & Mask':
            try:
                return np.array(Image.open(f'{self.mix_dir}{i:06d}.png'))
            except:
                return None
        
        if mode == 'Mask':
            try:
                return np.array(Image.open(f'{self.mask_dir}{i:06d}.png'))
            except:
               return None


class Tracker:
    def __init__(self, img, start, stop, device=DEVICE) -> None:
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
            'device': device,
        }
        aot_args = {
            'phase': 'PRE_YTB_DAV',
            'model': 'r50_deaotl',
            'model_path': 'ckpt/R50_DeAOTL_PRE_YTB_DAV.pth',
            'long_term_mem_gap': 250,
            'max_len_long_term': 250,
            'device': device,
        }
        segtracker_args = {
            'sam_gap': int(1e6), # the interval to run sam to segment new objects
            'min_area': 200, # minimal mask area to add a new mask as a new object
            'max_obj_num': 10, # maximal object number to track in a video
            'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80% 
        }
        self.first_frame = img
        self.start = start
        self.stop = stop
        self.segtracker = SegTracker(segtracker_args,sam_args,aot_args)
        self.segtracker.restart_tracker()
        self.coords = []
        self.modes = []

    def click(self, point, mode):
        self.coords.append(point)
        self.modes.append(mode)
        
        return self.inference()
    
    def inference(self):
        if len(self.coords) == 0:
            return None, None
        mask, mix = self.segtracker.seg_acc_click(self.first_frame, np.array(self.coords), np.array(self.modes))
        self.mask = mask
        return mask, mix

    
    def undo(self):
        self.coords = []
        self.modes = []

        return self.inference()

    def next_obj(self):
        if len(self.coords) == 0:
            return
        self.coords = []
        self.modes = []
        self.segtracker.update_origin_merged_mask(self.mask)
        self.segtracker.curr_idx += 1
    
    def tracking(self, v, progress):
        self.segtracker.add_reference(self.first_frame, self.mask, 0)
        self.segtracker.first_frame_mask = self.mask
        delta = 1 if self.start < self.stop else -1
        print(self.start, self.stop + delta, delta)
        for i in progress.tqdm(range(self.start, self.stop + delta, delta)):
            frame = np.array(Image.open(f'{v.frame_dir}{i:06d}.png'))
            
            if i == self.start:
                mask = self.segtracker.first_frame_mask
            else:
                mask = self.segtracker.track(frame, update_memory=True)
            

            mix = draw_mask(frame.copy(), mask)
            mask = build_putpalette(mask).convert(mode='RGB') # type: PIL.Image.Image
            mask = np.array(mask) # type: numpy.ndarray
            mask = cv2.cvtColor(mask,cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{v.mask_dir}{i:06d}.png', mask)
            
            mix = cv2.cvtColor(mix,cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{v.mix_dir}{i:06d}.png', mix)
        
        return "Track Finish"







def init_video(obj, input, progress=gr.Progress()):
    obj = Video_obj(input, progress)
    return obj, "video load ....ok"


def add_object_fn(obj, name):
    if len(name) == 0:
        return obj, "", f"Empty Object name"
    
    n = obj.add_object(name)
    return obj, "", f"Add Object, number of Object: {n}"


def obj_setting_done_fn(obj):
    return obj, gr.update(maximum=obj.n_frame-1), gr.update(maximum=obj.n_frame-1), gr.update(maximum=obj.n_frame-1), gr.update(visible=False), gr.update(visible=True)


def edit_tab_fn():
    txt = """
    Select Start & Stop Frame. 
    Label on start frame and track until stop frame."
    """
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), txt, True


def view_tab_fn():

    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), False

def frame_index_slide_fn(obj, index, mode):
    out = obj.display(int(index), mode)

    return out

def select_start_frame_slide_fn(v, tk, start, stop): #[l, r]
    ff = frame_index_slide_fn(v, start, "Image") # ff: First Frame
    tk = Tracker(ff, start, stop)
    return ff, tk

def select_stop_frame_slide_fn(tk, stop):
    if tk == None:
        return None, "Select Start frame first."
    tk.stop = stop
    return tk, "Stop frame Set"

def track_click_fn(tk, click_type_drop, ok, evt:gr.SelectData):
    if ok != True:
        return tk, None, "Not Edit Mode!"
    
    mode = 1 if click_type_drop == "Positive" else 0
    point = (evt.index[0], evt.index[1]) #(y, x)

    _, mix = tk.click(point, mode)

    return tk, mix, "click!"

def track_click_next_fn(tk):
    tk.next_obj()
    return tk, "Label next object."

def tracking_btm_fn(tk, v, progress=gr.Progress()):
    tk.tracking(v, progress)
    return tk, "Tracking Done"


def undo_btm_fn(tk):
    _, mix = tk.undo()

    return tk, mix, "Relabel this object"

def device_setting(device):
    DEVICE = device
    return f"Device: {device}."

def generate_output_btm_fn(v):
    n_mask = glob.glob(f'{v.mask_dir}*.png')
    n_mix = glob.glob(f'{v.mix_dir}*.png')

    print(n_mask, n_mix)
    if n_mask != v.n_frame or n_mix != v.n_frame:
        return  "Error", None, None, None

    mask_video = v.write_video('mask')
    mix_video = v.write_video('mix')
    return "OK", mask_video, mix_video, None





app = gr.Blocks()

with app:
    # Variable
    video_obj = gr.State(None)
    tracker_obj = gr.State(None)
    # Front-end
    gr.Markdown(HEADER)
    display_txt = gr.Textbox(label="Log")
    is_click = gr.State(None)

    with gr.Row(visible=True) as file_page:
        with gr.Column(scale=0.8):
            input_video = gr.File(label='Input video')
        with gr.Column(scale=0.2):
            with gr.Column():
                object_name_txt = gr.Textbox(label="Enter Object Name",
                                             interactive=True)
                add_object_btm = gr.Button("Add New Object",
                                           interactive=True)
                obj_setting_done_btm = gr.Button("Objects Setting Done",
                                                 interactive=True)

    with gr.Row(visible=False) as edit_page:
        with gr.Column(scale=0.8):

            display_img = gr.Image(label='Display', interactive=False)
            frame_index_slide = gr.Slider(label="Frame Index",
                                          minimum=0, step=1, maximum=10,
                                          value=0, interactive=True)

            select_start_frame_slide = gr.Slider(
                label="Enter Start Frame Index",
                minimum=0, step=1, maximum=10, value=0, interactive=True, visible=False)

            select_stop_frame_slide = gr.Slider(
                label="Enter Stop Frame Index",
                minimum=0, step=1, maximum=10, value=0, interactive=True, visible=False)

        with gr.Column(scale=0.2):
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
                    label="Device", value="cuda",
                    interactive=True)
                
                click_type_drop = gr.Dropdown(
                    choices=["Positive", "Negative"],
                    label="Click Type", value="Positive",
                    interactive=True)
                undo_btm = gr.Button("Undo", interactive=True)
                click_done_btm = gr.Button("Done", interactive=True)
                
                tracking_btm = gr.Button("Tracking", interactive=True)
                output_page_btm = gr.Button("Open Output Setting Page", interactive=True)

    with gr.Row(visible=False) as output_page:
        generate_output_btm = gr.Button("Output Video", interactive=True)
        mask_video = gr.File(label='Mask Video', interactive=False)
        mix_video = gr.File(label='Mask & Frame Video', interactive=False)
        object_config = gr.File(label='Object Config', interactive=False)

    # Function

    input_video.change(
        fn=init_video,
        inputs=[video_obj, input_video],
        outputs=[video_obj, display_txt],
    )

    add_object_btm.click(
        fn=add_object_fn,
        inputs=[video_obj, object_name_txt],
        outputs=[video_obj, object_name_txt, display_txt],
    )

    obj_setting_done_btm.click(
        fn=obj_setting_done_fn,
        inputs=[video_obj],
        outputs=[video_obj, frame_index_slide, select_start_frame_slide,
                 select_stop_frame_slide, file_page, edit_page],
    )

    edit_tab.select(
        fn=edit_tab_fn,
        outputs=[frame_index_slide, select_start_frame_slide,
                 select_stop_frame_slide, display_txt, is_click],

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

    select_start_frame_slide.change(
        fn=select_start_frame_slide_fn,
        inputs=[video_obj, tracker_obj, select_start_frame_slide, select_stop_frame_slide],
        outputs=[display_img, tracker_obj],
    )

    select_stop_frame_slide.change(
        fn=select_stop_frame_slide_fn,
        inputs=[tracker_obj, select_stop_frame_slide],
        outputs=[tracker_obj, display_txt],

    )



    device_drop.change(
        fn=device_setting,
        inputs=[device_drop],
        outputs=[display_txt],
    )
    

    display_img.select(
        fn=track_click_fn,
        inputs=[tracker_obj, click_type_drop, is_click],
        outputs=[tracker_obj, display_img, display_txt],
    )

    click_done_btm.click(
        fn=track_click_next_fn,
        inputs=[tracker_obj],
        outputs=[tracker_obj, display_txt],
    )

    undo_btm.click(
        fn=undo_btm_fn,
        inputs=[tracker_obj],
        outputs=[tracker_obj, display_img, display_txt],
    )


    tracking_btm.click(
        fn=tracking_btm_fn,
        inputs=[tracker_obj, video_obj],
        outputs=[tracker_obj, display_txt],
    )

    output_page_btm.click(
        fn=show_page,
        outputs=[output_page]
    )

    generate_output_btm.click(
        fn=generate_output_btm_fn,
        inputs=[video_obj],
        outputs=[display_txt, mask_video, mix_video, object_config]
    )

if __name__ == "__main__":
    app.queue(concurrency_count=5)
    app.launch(debug=True, share=False,
               server_name="0.0.0.0", server_port=10001).queue()
