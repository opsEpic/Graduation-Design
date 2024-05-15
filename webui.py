import json

import gradio as gr
import webbrowser

import numpy as np
import pandas as pd

import config
from trainer import *

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

if __name__ == '__main__':
    with gr.Blocks() as app:
        gr.Markdown('<h1>ECAPA_TDNN WebUI</h1>')
        gr.Markdown('NEU qjx 20206189')
        with gr.Row():
            with gr.Column():
                with gr.Tab('信息'):
                    gr.Markdown('<h2>config</h2>')
                    config_json = gr.Code(value=json.dumps(config.config, indent=4), language='json')

                    with gr.Row():
                        train_dataset_path = gr.Text(label='train_dataset_path', value=config.config['train_dataset_path'], interactive=True)
                        train_list_path = gr.Text(label='train_list_path', value=config.config['train_list_path'], interactive=True)
                    with gr.Row():
                        eval_dataset_path = gr.Text(label='eval_dataset_path', value=config.config['eval_dataset_path'], interactive=True)
                        eval_list_path = gr.Text(label='eval_list_path', value=config.config['eval_list_path'], interactive=True)
                    with gr.Row():
                        speaker = gr.Slider(label='speaker', value=config.config['speaker'], minimum=2, maximum=16384, step=1, interactive=True)
                        batch_size = gr.Slider(label='batch_size', value=config.config['batch_size'], minimum=8, maximum=2048, step=8, interactive=True)
                        slice_length = gr.Slider(label='slice_length', value=config.config['slice_length'], minimum=4000, maximum=80000, step=4000, interactive=True)
                    with gr.Row():
                        device = gr.Dropdown(label=f'device（推荐：{device}）', choices=['cuda', 'mps', 'cpu'], value=config.config['device'], interactive=True)
                        save_path = gr.Text(label='save_path', value=config.config['save_path'], interactive=True)


                    def config_update(key, val):
                        config.config[key] = val
                        config.save()
                        return json.dumps(config.config, indent=4)


                    train_dataset_path.change(config_update, inputs=[gr.Text(value='train_dataset_path', visible=False), train_dataset_path], outputs=[config_json])
                    train_list_path.change(config_update, inputs=[gr.Text(value='train_list_path', visible=False), train_list_path], outputs=[config_json])
                    eval_dataset_path.change(config_update, inputs=[gr.Text(value='eval_dataset_path', visible=False), eval_dataset_path], outputs=[config_json])
                    eval_list_path.change(config_update, inputs=[gr.Text(value='eval_list_path', visible=False), eval_list_path], outputs=[config_json])
                    speaker.change(config_update, inputs=[gr.Text(value='speaker', visible=False), speaker], outputs=[config_json])
                    batch_size.change(config_update, inputs=[gr.Text(value='batch_size', visible=False), batch_size], outputs=[config_json])
                    slice_length.change(config_update, inputs=[gr.Text(value='slice_length', visible=False), slice_length], outputs=[config_json])
                    device.change(config_update, inputs=[gr.Text(value='device', visible=False), device], outputs=[config_json])
                    save_path.change(config_update, inputs=[gr.Text(value='save_path', visible=False), save_path], outputs=[config_json])

            with gr.Column():
                with gr.Tab('训练'):
                    gr.Markdown('<h2>训练</h2>')
                    gr.Markdown('''`train_dataset_path`训练集目录。<br/>
                                `train_list_path`训练集列表地址。<br/>
                                `speaker`训练集说话人总数。<br/>
                                `batch_size`显存足够可以调大。<br/>
                                `slice_length`每个切片的采样数（时长 * 采样率）。<br/>
                                `device`训练设备。<br/>
                                `save_path`模型保存路径。<br/>''')

                    with gr.Row():
                        model_type1 = gr.Dropdown(label='模型类型', choices=['ECAPA_TDNN', 'PCF_ECAPA'], value='ECAPA_TDNN', interactive=True)
                    with gr.Row():
                        epoch = gr.Slider(label='训练轮数', value=1, minimum=1, maximum=1024, step=1, interactive=True)
                    with gr.Row():
                        model_train = gr.Button(value='开始训练')


                    def func_model_train(in_type, in_epoch):
                        types = {
                            'ECAPA_TDNN': ECAPATDNN,
                            'PCF_ECAPA': PCFECAPATDNN
                        }

                        model = Modeler(types[in_type](), config.config['speaker'], config.config['device'], last_model)
                        for _ in range(in_epoch):
                            model.model_train(trainloader)
                            model.model_save(config.config['save_path'])
                            time.sleep(100)


                    model_train.click(func_model_train, inputs=[model_type1, epoch])

                with gr.Tab('验证'):
                    gr.Markdown('<h2>验证</h2>')
                    gr.Markdown('''`eval_dataset_path`验证集目录。<br/>
                                `eval_list_path`验证集列表地址。<br/>
                                `slice_length`每个切片的采样数（时长 * 采样率）。<br/>
                                `device`验证设备。<br/>
                                `save_path`模型保存路径。<br/>''')

                    with gr.Row():
                        model_type2 = gr.Dropdown(label='模型类型', choices=['ECAPA_TDNN', 'PCF_ECAPA'], value='ECAPA_TDNN', interactive=True)
                    with gr.Row():
                        target1 = gr.File(label='目标模型', file_count='multiple', file_types=['.pt'])
                    with gr.Row():
                        model_eval = gr.Button(value='开始验证')


                    def func_model_eval(in_type, in_models):
                        types = {
                            'ECAPA_TDNN': ECAPATDNN,
                            'PCF_ECAPA': PCFECAPATDNN
                        }

                        for in_model in in_models:
                            model = Modeler(types[in_type](), config.config['speaker'], config.config['device'], in_model)
                            print(model.model_eval(config.config['eval_list_path'], config.config['eval_dataset_path'], config.config['slice_length']))
                            time.sleep(100)


                    model_eval.click(func_model_eval, inputs=[model_type2, target1])

                with gr.Tab('推理'):
                    gr.Markdown('<h2>推理</h2>')
                    gr.Markdown('''`slice_length`每个切片的采样数（时长 * 采样率）。<br/>
                                `device`推理设备。<br/>''')
                    with gr.Row():
                        model_type3 = gr.Dropdown(label='模型类型', choices=['ECAPA_TDNN', 'PCF_ECAPA'], value='ECAPA_TDNN', interactive=True)
                    with gr.Row():
                        audio1 = gr.Audio(label='文件或录制')
                        audio2 = gr.Audio(label='文件或录制')
                    with gr.Row():
                        target2 = gr.File(label='目标模型', file_count='single', file_types=['.pt'])
                    with gr.Row():
                        infer = gr.Button(value='开始推理')
                    with gr.Row():
                        plot = gr.BarPlot(
                            pd.DataFrame({'x': [''], 'y': [0]}),
                            x='x',
                            y='y',
                            vertical=False,
                            y_lim=[-1, 1],
                            x_title='',
                            y_title='',
                            title='相似度',
                        )


                    def func_model_infer(in_type, in_model, in_audio1, in_audio2):
                        types = {
                            'ECAPA_TDNN': ECAPATDNN,
                            'PCF_ECAPA': PCFECAPATDNN
                        }

                        in_audio1 = np.array(in_audio1[1], dtype=np.float32)
                        in_audio1 /= np.max(in_audio1)

                        in_audio2 = np.array(in_audio2[1], dtype=np.float32)
                        in_audio2 /= np.max(in_audio2)

                        model = Modeler(types[in_type](), config.config['speaker'], config.config['device'], in_model)
                        score = model.model_infer([in_audio1, in_audio2], config.config['slice_length'])
                        return pd.DataFrame({'x': [''], 'y': [score]})


                    infer.click(func_model_infer, inputs=[model_type3, target2, audio1, audio2], outputs=[plot])

    port = 4567
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.launch(server_port=port)
