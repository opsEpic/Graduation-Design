import json
import logging

import torch
import numpy as np
import gradio as gr
import webbrowser
import librosa

from trainer import Trainer

recommend_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = Trainer()
infer_stream_buffer = []


def get_json_config():
    return json.dumps(model.config.config, indent=4)


def config_save():
    model.config.save()


def config_update(key, val):
    model.config.config[key] = val
    return get_json_config()


def model_pretrain():
    model.model_pretrain()
    return '预处理完成'


def model_load_dataset():
    model.load_dataset()
    return '训练集加载完成'


def model_load_model():
    model.load_model()
    return '模型加载完成', str(model.model.step)


def model_train():
    model.model_train()
    return '训练完成', str(model.model.step)


def model_test():
    model.model_test()
    return '测试完成'


def model_infer(wav):
    sr, audio = wav
    audio = np.array(audio, dtype=np.float32)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    label = model.model_infer(audio)
    return str(label)


def model_infer_stream(wav, buffer_size, old_news):
    global infer_stream_buffer

    sr, audio = wav
    audio = np.array(audio, dtype=np.float32)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    infer_stream_buffer.extend(audio)

    buffer_size = int(buffer_size)
    if len(infer_stream_buffer) > buffer_size:
        label = model.model_infer(infer_stream_buffer[:buffer_size])
        infer_stream_buffer = []
        return str(label)

    return old_news


def main():
    with gr.Blocks() as app:
        with gr.Tab('ECAPA_TDNN'):
            gr.Markdown('<h1>ECAPA_TDNN WebUI</h1>')
            gr.Markdown('NEU qjx 20206189')
            with gr.Row():
                with gr.Column():
                    with gr.Tab('信息'):
                        gr.Markdown('<h2>config</h2>')
                        config = gr.Code(value=get_json_config(), language='json')
                        save = gr.Button(value='保存（点击才能保存到文件）')

                        save.click(config_save)

                        gr.Markdown('<h2>state</h2>')
                        with gr.Row():
                            load_dataset = gr.Button(value='加载训练集')
                            load_model = gr.Button(value='加载模型')
                        step = gr.Text(label='模型训练步数', value='0')
                        test = gr.Button(value='开始测试')
                        text_output = gr.Text(label='output')

                        load_dataset.click(model_load_dataset, outputs=[text_output])
                        load_model.click(model_load_model, outputs=[text_output, step])
                        test.click(model_test, outputs=[text_output])

                with gr.Column():
                    with gr.Tab('预处理'):
                        gr.Markdown('<h2>预处理</h2>')
                        gr.Markdown('`dataset_path`数据集路径，支持 voxceleb 或目录结构与其类似的数据集。<br/>'
                                    '`eval_list_size`验证集音频数。<br/>'
                                    '`eval_list_size`测试集音频数。<br/>'
                                    '`train_list_path`训练集file_list输出路径。<br/>'
                                    '`eval_list_path`验证集file_list输出路径。<br/>'
                                    '`test_list_path`测试集file_list输出路径。<br/>')

                        dataset_path = gr.Text(label='dataset_path', value=model.config.config['dataset_path'], interactive=True)
                        with gr.Row():
                            eval_list_size = gr.Slider(label='eval_list_size', value=model.config.config['eval_list_size'], minimum=1, maximum=512, step=1, interactive=True)
                            test_list_size = gr.Slider(label='test_list_size', value=model.config.config['test_list_size'], minimum=1, maximum=512, step=1, interactive=True)
                        train_list_path = gr.Text(label='train_list_path', value=model.config.config['train_list_path'], interactive=True)
                        eval_list_path = gr.Text(label='eval_list_path', value=model.config.config['eval_list_path'], interactive=True)
                        test_list_path = gr.Text(label='test_list_path', value=model.config.config['test_list_path'], interactive=True)
                        pretrain = gr.Button(value='开始预处理')
                        pretrain_output = gr.Text(label='output', value='')

                        dataset_path.change(config_update, inputs=[gr.Text(value='dataset_path', visible=False), dataset_path], outputs=[config])
                        eval_list_size.change(config_update, inputs=[gr.Text(value='eval_list_size', visible=False), eval_list_size], outputs=[config])
                        test_list_size.change(config_update, inputs=[gr.Text(value='test_list_size', visible=False), test_list_size], outputs=[config])
                        train_list_path.change(config_update, inputs=[gr.Text(value='train_list_path', visible=False), train_list_path], outputs=[config])
                        eval_list_path.change(config_update, inputs=[gr.Text(value='eval_list_path', visible=False), eval_list_path], outputs=[config])
                        test_list_path.change(config_update, inputs=[gr.Text(value='test_list_path', visible=False), test_list_path], outputs=[config])
                        pretrain.click(model_pretrain, outputs=[pretrain_output])

                    with gr.Tab('训练'):
                        gr.Markdown('<h2>训练</h2>')
                        gr.Markdown('`C`模型中间层channels大小，默认512。<br/>'
                                    '`speaker`训练集说话人总数。<br/>'
                                    '`batch_size`内存足够可以调大。<br/>'
                                    '`slice_length`每个切片的采样数（时长 * 采样率）。<br/>'
                                    '`model_train_epoch`模型训练轮数。<br/>'
                                    '`device`训练设备。<br/>'
                                    '`model_save`模型保存与否。<br/>'
                                    '`model_save_path`模型保存路径。<br/>')

                        with gr.Row():
                            C = gr.Slider(label='C', value=model.config.config['C'], minimum=128, maximum=2048, step=8, interactive=True)
                            speaker = gr.Slider(label='speaker', value=model.config.config['speaker'], minimum=2, maximum=16384, step=1, interactive=True)
                        with gr.Row():
                            batch_size = gr.Slider(label='batch_size', value=model.config.config['batch_size'], minimum=8, maximum=2048, step=8, interactive=True)
                            slice_length = gr.Slider(label='slice_length', value=model.config.config['slice_length'], minimum=8000, maximum=160000, step=1000, interactive=True)
                        with gr.Row():
                            model_train_epoch = gr.Slider(label='model_train_epoch', value=model.config.config['model_train_epoch'], minimum=1, maximum=1024, step=1, interactive=True)
                            device = gr.Dropdown(label=f'device（推荐：{recommend_device}）', choices=['cuda', 'mps', 'cpu'], value=model.config.config['device'], interactive=True)
                        model_save = gr.Checkbox(label='model_save', value=model.config.config['model_save'], interactive=True)
                        model_save_path = gr.Text(label='model_save_path', value=model.config.config['model_save_path'], interactive=True)
                        train = gr.Button(value='开始训练')
                        train_output = gr.Text(label='output', value='')

                        C.change(config_update, inputs=[gr.Text(value='C', visible=False), C], outputs=[config])
                        speaker.change(config_update, inputs=[gr.Text(value='speaker', visible=False), speaker], outputs=[config])
                        batch_size.change(config_update, inputs=[gr.Text(value='batch_size', visible=False), batch_size], outputs=[config])
                        slice_length.change(config_update, inputs=[gr.Text(value='slice_length', visible=False), slice_length], outputs=[config])
                        model_train_epoch.change(config_update, inputs=[gr.Text(value='model_train_epoch', visible=False), model_train_epoch], outputs=[config])
                        device.change(config_update, inputs=[gr.Text(value='device', visible=False), device], outputs=[config])
                        model_save.change(config_update, inputs=[gr.Text(value='model_save', visible=False), model_save], outputs=[config])
                        model_save_path.change(config_update, inputs=[gr.Text(value='model_save_path', visible=False), model_save_path], outputs=[config])
                        train.click(model_train, outputs=[train_output, step])

                    with gr.Tab('推理'):
                        gr.Markdown('<h2>流式推理</h2>')

                        buffer_size = gr.Slider(label='流式推理缓冲大小（采样数）', value=16000, minimum=4000, maximum=80000, step=4000, interactive=True)
                        audio_stream = gr.Audio(label='流式推理', streaming=True)
                        infer_output_stream = gr.Text(label='output', value='')

                        audio_stream.stream(model_infer_stream, inputs=[audio_stream, buffer_size, infer_output_stream], outputs=[infer_output_stream])

                        gr.Markdown('<h2>流式推理</h2>')

                        audio = gr.Audio(label='文件或录制推理')
                        infer = gr.Button(value='开始推理')
                        infer_output = gr.Text(label='output', value='')

                        infer.click(model_infer, inputs=[audio], outputs=[infer_output])

    port = 4567
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.launch(server_port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
