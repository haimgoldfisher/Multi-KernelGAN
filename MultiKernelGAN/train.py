import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
import torch
import re

def extract_numbers_from_filename(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return None

def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        if iteration == 0:
            img_name = os.path.basename(conf.input_image_path)
            img_num = extract_numbers_from_filename(img_name)
            weights_path = conf.weights_dir_path
            file_path = os.path.join(weights_path, "img"+str(img_num) + "/gan_checkpoints/model.pt")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            print("\nfile_path = ", file_path)
            if os.path.isfile(file_path):
                print("File exists!")
                # load weights
                G_optimizer = gan.optimizer_G
                D_optimizer = gan.optimizer_D
                checkpoint = torch.load(file_path)
                gan.G.load_state_dict(checkpoint['G_state_dict'])
                gan.D.load_state_dict(checkpoint['D_state_dict'])
                G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
                D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
                print("@@@@@ gan weights loaded @@@@")
            else:
                print("File does not exist.")
                # save weights
                torch.save({
                    'G_state_dict': gan.G.state_dict(),
                    'D_state_dict': gan.D.state_dict(),
                    'G_optimizer_state_dict': gan.optimizer_G.state_dict(),
                    'D_optimizer_state_dict': gan.optimizer_D.state_dict(),
                }, file_path)
                print("@@@@@ gan weights saved @@@@")
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    gan.finish()

def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--weights-dir', type=str, default='weights', help='path to weights directory.')
    prog.add_argument('--masks-dir', type=str, default='masks', help='path to masks directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)

def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale),
              '--weights_dir_path', args.weights_dir ,
              '--masks_dir_path', args.masks_dir]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    return params

if __name__ == '__main__':
    main()
