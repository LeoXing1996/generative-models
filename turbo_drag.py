from argparse import ArgumentParser

import gradio as gr

from utils import (drag_image, generate_image, get_points, store_img_gen,
                   undo_points)

parser = ArgumentParser()
parser.add_argument('--server-name', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--share', action='store_true')

parser.add_argument('--is-debug', action='store_true')
parser.add_argument('--save-path', default='work_dirs')

args = parser.parse_args()


LENGTH = 480


with gr.Blocks() as app:
    mask_gen = gr.State(value=None)  # store mask
    selected_points_gen = gr.State([])  # store points
    # store the diffusion-generated image
    original_image_gen = gr.State(value=None)
    # store the intermediate diffusion latent during generation
    init_latent = gr.State(value=None)
    pipeline = gr.State(value=None)
    save_path = gr.State(value=args.save_path)
    is_debug = gr.State(value=args.is_debug)

    # general parameters
    # with gr.Row():
    pos_prompt_gen = gr.Textbox(label="Positive Prompt")
    # image = gr.Image(label='Generated Image')
    gen_img_button = gr.Button("Generate Image")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
            # canvas_gen = gr.ImageMask(type="numpy", tool="sketch", label="Draw Mask",
            #                           show_label=True, height=LENGTH, width=LENGTH)  # for mask painting
            canvas_gen = gr.ImageMask(type="numpy", tool="sketch", label="Draw Mask",
                                      show_label=True, height=LENGTH)  # for mask painting
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Click Points</p>""")
            input_image_gen = gr.Image(type="numpy", label="Click Points",
                                       show_label=True, height=LENGTH, width=LENGTH)  # for points clicking
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Editing Results</p>""")
            output_image_gen = gr.Image(type="numpy", label="Editing Results",
                                        show_label=True, height=LENGTH, width=LENGTH)
            # output_image_gen = gr.Video(label='Editing Results',
            #                             show_label=True, height=LENGTH, width=LENGTH)

    with gr.Row():
        undo_button_gen = gr.Button("Undo point")
        run_button_gen = gr.Button("Run")
        clear_all_button_gen = gr.Button("Clear All")

    with gr.Tab(label="Drag Config"):
        with gr.Row():
            n_pix_step_gen = gr.Number(
                value=40,
                label="Number of Pixel Steps",
                info="Number of gradient descent (motion supervision) steps on latent.",
                precision=0)
            lam_gen = gr.Number(value=0.01, label="lam",
                                info="regularization strength on unmasked areas")
            latent_lr_gen = gr.Number(value=0.05, label="latent lr")
            # mask: 0.1 -> 0.01
            # lr: 0.01 -> 0.05
            # start_step_gen = gr.Number(
            #     value=0, label="start_step", precision=0, visible=False)
            # start_layer_gen = gr.Number(
            #     value=10, label="start_layer", precision=0, visible=False)

    gen_img_button.click(
        generate_image,
        inputs=[pos_prompt_gen, pipeline],
        outputs=[canvas_gen, init_latent, pipeline])

    canvas_gen.edit(
        store_img_gen,
        inputs=[canvas_gen],
        outputs=[original_image_gen, selected_points_gen,
                 input_image_gen, mask_gen]
    )
    input_image_gen.select(
        get_points,
        inputs=[input_image_gen, selected_points_gen],
        outputs=[input_image_gen],
    )
    # TODO: convert intermedia feature to init latent
    run_button_gen.click(
        drag_image,
        inputs=[pipeline,
                original_image_gen,
                pos_prompt_gen,
                selected_points_gen,
                mask_gen,
                init_latent,
                latent_lr_gen,
                n_pix_step_gen,
                lam_gen,
                save_path,
                is_debug,
            ],
        outputs=[
            init_latent,
            output_image_gen
        ]
    )


app.launch(server_name=args.server_name,
           server_port=args.port, share=args.share)
