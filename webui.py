import json
import logging

import torch
import gradio as gr
import webbrowser

from trainer import Trainer

recommend_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = Trainer()


def get_json_config():
    return json.dumps(model.config.config, indent=4)


def config_update(key, val):
    model.config.config[key] = val
    return get_json_config()


def model_pretrain():
    model.config.save()
    model.model_pretrain()
    return '保存并预处理完成'


def model_load():
    model.load_dataset()
    model.load_model()
    return '加载完成', '已加载', str(model.model.step)


def model_train():
    model.config.save()
    model.model_train()
    return '保存并训练完成', str(model.model.step)


def model_test():
    model.model_test()
    return '测试完成'


# todo
def model_infer():
    pass


def main():
    with gr.Blocks() as app:
        gr.Markdown('<h1>ECAPA_TDNN WebUI</h1>')
        gr.Markdown('NEU qjx 20206189')
        with gr.Row():
            with gr.Column():
                with gr.Tab('配置'):
                    gr.Markdown('<h2>config:</h2>')
                    gr.Markdown('此处更新不会立即保存到./config/train_config.json文件，')
                    gr.Markdown('点击任意按钮完成操作同时保存。')
                    config = gr.Code(value=get_json_config(), language='json')
                with gr.Tab('模型状态'):
                    gr.Markdown('<h2>model state:</h2>')
                    state = gr.Text(label='状态', value='未加载')
                    step = gr.Text(label='模型训练步数', value='0')
                    test = gr.Button(value='开始测试')
                    text_output = gr.Text(label='output')

                    test.click(model_test, outputs=[text_output])

            with gr.Column():
                with gr.Tab('预处理'):
                    gr.Markdown('<h2>预处理</h2>')
                    gr.Markdown('`dataset_path`数据集路径，仅支持 voxceleb 或目录结构与其相同的数据集。<br/>'
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
                    pretrain = gr.Button(value='保存并开始预处理')
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
                                '`slice_length`每个切片的采样数(时长 * 采样率)。<br/>'
                                '`model_train_epoch`模型训练轮数。<br/>'
                                '`device`训练设备。<br/>'
                                '`model_save`模型保存与否。<br/>'
                                '`model_save_path`模型保存路径。<br/>')

                    with gr.Row():
                        C = gr.Slider(label='C', value=model.config.config['C'], minimum=128, maximum=2048, step=8, interactive=True)
                        speaker = gr.Slider(label='speaker', value=model.config.config['speaker'], minimum=1, maximum=16384, step=1, interactive=True)
                    with gr.Row():
                        batch_size = gr.Slider(label='batch_size', value=model.config.config['batch_size'], minimum=8, maximum=2048, step=8, interactive=True)
                        slice_length = gr.Slider(label='slice_length', value=model.config.config['slice_length'], minimum=8000, maximum=160000, step=1000, interactive=True)
                    with gr.Row():
                        model_train_epoch = gr.Slider(label='model_train_epoch', value=model.config.config['model_train_epoch'], minimum=1, maximum=1024, step=1, interactive=True)
                        device = gr.Dropdown(label=f'device(推荐：{recommend_device})', choices=['cuda', 'mps', 'cpu'], value=model.config.config['device'], interactive=True)
                    model_save = gr.Checkbox(label='model_save', value=model.config.config['model_save'], interactive=True)
                    model_save_path = gr.Text(label='model_save_path', value=model.config.config['model_save_path'], interactive=True)
                    load = gr.Button(value='加载训练集与模型')
                    train = gr.Button(value='保存并开始训练')
                    train_output = gr.Text(label='output', value='')

                    C.change(config_update, inputs=[gr.Text(value='C', visible=False), C], outputs=[config])
                    speaker.change(config_update, inputs=[gr.Text(value='speaker', visible=False), speaker], outputs=[config])
                    batch_size.change(config_update, inputs=[gr.Text(value='batch_size', visible=False), batch_size], outputs=[config])
                    slice_length.change(config_update, inputs=[gr.Text(value='slice_length', visible=False), slice_length], outputs=[config])
                    model_train_epoch.change(config_update, inputs=[gr.Text(value='model_train_epoch', visible=False), model_train_epoch], outputs=[config])
                    device.change(config_update, inputs=[gr.Text(value='device', visible=False), device], outputs=[config])
                    model_save.change(config_update, inputs=[gr.Text(value='model_save', visible=False), model_save], outputs=[config])
                    model_save_path.change(config_update, inputs=[gr.Text(value='model_save_path', visible=False), model_save_path], outputs=[config])
                    load.click(model_load, outputs=[train_output, state, step])
                    train.click(model_train, outputs=[train_output, step])

    print("推理页面已开启!")
    port = 4567
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.launch(server_port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
