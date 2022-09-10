import argparse
import train
import test
import os
import ast


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=ast.literal_eval, default=None)
    parser.add_argument("--use_cuda", type=ast.literal_eval, default=None)
    parser.add_argument("--seed", type=int, default=2022)

    parser.add_argument("--trainset", type=str, default="./SICE_subset/train_data/")
    parser.add_argument("--testset", type=str, default="./SICE_subset/val_data/")

    parser.add_argument('--ckpt_path', default='./checkpoint/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument('--fused_img_path', default='./fused_result/', type=str,
                        metavar='PATH', help='path to save images')
    parser.add_argument('--model_path', type=str, default='./model/',
                    help='trained model directory')

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--decay_interval", type=int, default=50)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--epochs_per_eval", type=int, default=10)#
    parser.add_argument("--epochs_per_save", type=int, default=10)#

    return parser.parse_args()


def main(cfg):
    if cfg.train:
        t = train.Trainer(cfg)
        t.fit()
    else:
        t = test.Test(cfg)
        t.test()


if __name__ == "__main__":
    config = parse_config()

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.fused_img_path):
        os.makedirs(config.fused_img_path)
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    main(config)
