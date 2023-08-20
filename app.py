HEADER = """
<div style="text-align:center;">
    <span style="font-size:3em; font-weight:bold;">Tracking Gold Apple</span>
</div>
"""
import gradio as gr


def show_edit_page():
    print("off")
    return gr.update(visible=True)


app = gr.Blocks()
with app:
    # Variable
    gr.Markdown(HEADER)

    # Front-end
    with gr.Row():
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
            display_txt = gr.Textbox(label="Log")
        with gr.Column(scale=0.2):
            with gr.Tab(label="Edit"):

                select_start_frame_txt = gr.Textbox(
                    label="Enter Start Frame Index")
                select_stop_frame_txt = gr.Textbox(
                    label="Enter Stop Frame Index")
                select_frame_done_btm = gr.Button("Done", interactive=True)
                click_type_drop = gr.Dropdown(
                    choices=["Positive", "Negative"],
                    label="Click Type", value="Positive",
                    interactive=True)
                click_done_btm = gr.Button("Done", interactive=True)
                tracking_btm = gr.Button("Tracking", interactive=True)
            with gr.Tab(label="View"):
                display_mode_drop = gr.Dropdown(
                    choices=["Image", "Image & Mask", "Mask"],
                    label="Display Mode", value="Image",
                    interactive=True)
                frame_index_slide = gr.Slider(label="Frame Index",
                                              minimum=0, step=1, maximum=10,
                                              value=0, interactive=True)
                step_frame_num_text = gr.Dropdown(
                    choices=["1", "5", "10", "100", "1000", "10000"],
                    label="Frame step number", value="1",
                    interactive=True)

    obj_setting_done_btm.click(
        fn=show_edit_page,
        outputs=edit_page,
    )


if __name__ == "__main__":
    app.queue(concurrency_count=5)
    app.launch(debug=True, share=False,
               server_name="0.0.0.0", server_port=10001).queue()
