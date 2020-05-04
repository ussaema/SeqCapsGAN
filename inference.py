import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from core.generator import Generator
from core.discriminator import Discriminator
from core.gan import GAN
import pickle
import argparse

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--word_to_idx_dir', type=str, required=True)

parser.add_argument('--image', type=str, required=True)
parser.add_argument('--max_length', type=int, default=25)

parser.add_argument('--load_model_dir', type=str, required=True)


args = parser.parse_args()

def main():

    # load vocabulary
    with open(args.word_to_idx_dir, 'rb') as f:
        word_to_idx = pickle.load(f)

    # graph session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # generator network
    generator = Generator(sess, word_to_idx, dim_embed=512, dim_hidden=1024,
                          n_time_step=args.max_length, prev2out=True, ctx2out=True, emo2out=True, alpha_c=1.0,
                          selector=True, dropout=True, features_extractor='vgg', update_rule='adam', pretrained_model=args.load_model_dir)

    generator.inference(args.image)


if __name__ == "__main__":
    main()