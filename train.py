import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from core.generator import Generator
from core.discriminator import Discriminator
from core.gan import GAN
from core.utils import load_coco_data, combine_vocab



def main():
    # load dataset
    train_data = load_coco_data(image_dir='data/images/train2014_resized/', caption_file='data/annotations/captions_train2014.json', splits=[0.001], max_length=20)
    val_data, test_data = load_coco_data(image_dir='data/images/val2014_resized/', caption_file='data/annotations/captions_val2014.json', splits=[0.01, 0.01], max_length=20)
    word_to_idx = combine_vocab([train_data['word_to_idx'], val_data['word_to_idx'], test_data['word_to_idx']])

    generator = Generator(word_to_idx, dim_embed=512, dim_hidden=1024,
                             n_time_step=20, prev2out=True, ctx2out=True, emo2out=True, alpha_c=1.0,
                             selector=True, dropout=True, features_extractor='vgg', update_rule='adam', learning_rate=0.001)

    discriminator = Discriminator(sequence_length=generator.T, num_classes=2, vocab_size=generator.V,
                                  embedding_size=512,
                                  filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16],
                                  num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                                  l2_reg_lambda=0.2)

    gan = GAN(generator, discriminator, pretrained_model=None, dis_dropout_keep_prob=1.0)


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    generator.train(sess, train_data, val_data, n_epochs=1, batch_size=8,
                    pretrained_model=None, save_every=10000, model_path='model/generator/',
                    validation=True, log_path='log/generator/', log_every=1)

    discriminator.train(sess, data=train_data, val_data=val_data, generator=generator, n_epochs=1, batch_size=4,
                        validation= True, dropout_keep_prob=0.75, iterations=1,
                        save_every=10000, log_every= 1, model_path='model/discriminator/',
                        log_path='log/discriminator/', pretrained_model=None)

    gan.train(sess, train_data, val_data, n_epochs=20, batch_size=8, rollout_num=5,
              validation=True, log_every=1, save_every=1,
              model_path='model/gan/', log_path='log/gan/')

if __name__ == "__main__":
    main()