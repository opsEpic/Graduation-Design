import ffmpeg
import glob
import os
from tqdm import tqdm


def trans(data_path, out_data_path):
    for file1 in tqdm(os.listdir(data_path)):
        for file2 in glob.glob(os.path.join(file1, '*/*.m4a'), root_dir=data_path):
            in_file = os.path.join(data_path, file2)
            out_file = os.path.join(out_data_path, file2[:-4] + '.wav')
            if not os.path.exists(out_file):
                out_path = os.path.dirname(out_file)
                os.makedirs(out_path, exist_ok=True)

                ffmpeg.input(in_file).output(out_file).global_args('-loglevel', 'quiet').run()
