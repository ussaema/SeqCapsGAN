import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from core.generator import Generator
from core.discriminator import Discriminator
from core.gan import GAN
import pickle
import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gen_train', action="store_true", default=False)
parser.add_argument('--disc_train', action="store_true", default=False)
parser.add_argument('--gan_train', action="store_true", default=False)

parser.add_argument('--gen_validate', action="store_true", default=False)
parser.add_argument('--disc_validate', action="store_true", default=False)
parser.add_argument('--gan_validate', action="store_true", default=False)

parser.add_argument('--word_to_idx_dir', type=str, default='data/word_to_idx.pkl')
parser.add_argument('--train_senticap_data_dir', type=str, default='data/train_senticap_data.pkl')
parser.add_argument('--val_senticap_data_dir', type=str, default='data/val_senticap_data.pkl')
parser.add_argument('--train_coco_data_dir', type=str, default='data/train_coco_data.pkl')
parser.add_argument('--val_coco_data_dir', type=str, default='data/val_coco_data.pkl')

parser.add_argument('--max_length', type=int, default=25)

parser.add_argument('--disc_network', type=str, default='capsnet')

parser.add_argument('--gen_load_model_dir', type=str, default=None)
parser.add_argument('--disc_load_model_dir', type=str, default=None)
parser.add_argument('--gan_load_model_dir', type=str, default=None)
parser.add_argument('--gen_save_model_dir', type=str, default='./model/generator/')
parser.add_argument('--disc_save_model_dir', type=str, default='./model/discriminator/')
parser.add_argument('--gan_save_model_dir', type=str, default='./model/gan/')
parser.add_argument('--gen_log_dir', type=str, default='./log/generator/')
parser.add_argument('--disc_log_dir', type=str, default='./log/discriminator/')
parser.add_argument('--gan_log_dir', type=str, default='./log/gan/')

parser.add_argument('--gen_dataset', type=str, default='coco')
parser.add_argument('--disc_dataset', type=str, default='coco')
parser.add_argument('--gan_dataset', type=str, default='senticap')

parser.add_argument('--batchsize', type=int, default=8)

parser.add_argument('--gen_epochs', type=int, default=10)
parser.add_argument('--disc_epochs', type=int, default=10)
parser.add_argument('--gan_epochs', type=int, default=10)

parser.add_argument('--gen_iters', type=int, default=1)
parser.add_argument('--disc_iters', type=int, default=1)

parser.add_argument('--gen_lr', type=float, default=0.001)
parser.add_argument('--disc_lr', type=float, default=0.0001)

args = parser.parse_args()

def main():

    # load vocabulary
    with open(args.word_to_idx_dir, 'rb') as f:
        word_to_idx = pickle.load(f)
    # load data
    with open(args.train_senticap_data_dir, 'rb') as f:
        train_senticap_data = pickle.load(f)
    with open(args.val_senticap_data_dir, 'rb') as f:
        val_senticap_data = pickle.load(f)
    with open(args.train_coco_data_dir, 'rb') as f:
        train_coco_data = pickle.load(f)
    with open(args.val_coco_data_dir, 'rb') as f:
        val_coco_data = pickle.load(f)

    # graph session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # generator network
    generator = Generator(sess, word_to_idx, dim_embed=512, dim_hidden=1024,
                          n_time_step=args.max_length, prev2out=True, ctx2out=True, emo2out=True, alpha_c=1.0,
                          selector=True, dropout=True, features_extractor='vgg', update_rule='adam',
                          learning_rate=args.gen_lr, pretrained_model=args.gen_load_model_dir)
    print("*" * 16, "Generator built", "*" * 16)
    if args.gen_train:
        # pre-train generator
        print('*' * 20 + "Start Training Generator" + '*' * 20)
        generator.train(train_senticap_data if args.gen_dataset == 'senticap' else train_coco_data, val_senticap_data if args.gen_dataset == 'senticap' else val_coco_data, n_epochs=args.gen_epochs, batch_size=args.batchsize,
                        save_every=1, model_path=args.gen_save_model_dir,
                        validation=args.gen_validate, log_path=args.gen_log_dir, log_every=1)

    # discriminator network
    discriminator = Discriminator(sess, sequence_length=generator.T, num_classes=2, vocab_size=generator.V,
                                  embedding_size=512,
                                  filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, args.max_length - 4],
                                  num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                                  l2_reg_lambda=0.2, pretrained_model=args.disc_load_model_dir, learning_rate=args.disc_lr, batch_size=args.batchsize, model=args.disc_network)
    print("*" * 16, "Discriminator built", "*" * 16)
    if args.disc_train:
        # pre-train discriminator
        print('*' * 20 + "Start Training Discriminator" + '*' * 20)
        discriminator.train(data=train_senticap_data if args.disc_dataset == 'senticap' else train_coco_data, val_data=val_senticap_data if args.disc_dataset == 'senticap' else val_coco_data, generator=generator, n_epochs=args.disc_epochs,
                            batch_size=args.batchsize,
                            validation=args.disc_validate, dropout_keep_prob=0.75, iterations=1,
                            save_every=1, log_every=1, model_path=args.disc_save_model_dir,
                            log_path=args.disc_log_dir)

    # gan network
    gan = GAN(sess, generator, discriminator, pretrained_model=args.gan_load_model_dir, dis_dropout_keep_prob=1.0)
    print("*" * 16, "GAN built", "*" * 16)
    if args.gan_train:
        # train gan
        print('*' * 20 + "Start Training GAN" + '*' * 20)
        gan.train(train_coco_data if args.gan_dataset == 'coco' else train_senticap_data, val_coco_data if args.gan_dataset == 'coco' else val_senticap_data, n_epochs=args.gan_epochs, batch_size=args.batchsize, rollout_num=5,
                  validation=args.gan_validate, log_every=1, save_every=1, gen_iterations=args.gen_iters, disc_iterations=args.disc_iters,
                  model_path=args.gan_save_model_dir, log_path=args.gan_log_dir)

if __name__ == "__main__":
    main()