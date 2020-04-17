import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import sys

from core.utils import *
from core.bleu import evaluate
import numpy as np
from core.log import *
from datetime import datetime
from tqdm import tqdm

class GAN(object):
    def __init__(self, generator, discriminator, pretrained_model=None, dis_dropout_keep_prob=1.):

        self.generator = generator
        self.discriminator = discriminator
        self.pretrained_model = pretrained_model
        self.dis_dropout_keep_prob = dis_dropout_keep_prob

        self.prev_gen_loss = -1
        self.prev_gen_scores = -1
        self.prev_disc_acc = -1
        self.prev_disc_loss = -1

    def get_reward(self, sess, features_batch, emotions_batch, captions_batch, rollout_num):
        rewards = []
        for i in range(rollout_num):
            # given_num between 1 to sequence_length - 1 for a part completed sentence
            for seq_length in range(1, self.generator.T):
                feed_dict = {self.generator.features: features_batch, self.generator.emotions: emotions_batch, self.generator.seq_length: seq_length}
                generated_captions = sess.run(self.generator.generated_captions, feed_dict)
                feed_dict = {self.discriminator.input_x: generated_captions, self.discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(self.discriminator.ypred_for_auc, feed_dict)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[seq_length - 1] += ypred

            # the last token reward
            feed_dict = {self.discriminator.input_x: captions_batch, self.discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(self.discriminator.ypred_for_auc, feed_dict)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                # completed sentence reward
                rewards[self.generator.T - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def train(self, data, val_data, n_epochs=10, batch_size=100, dis_batch_size=30, rollout_num=10, validation=False, gen_scores=['Bleu_1','Bleu_2','Bleu_3','Bleu_4', 'ROUGE_L', 'CIDEr'], log_every=100, save_every=1, log_path='./log/', model_path='./model/', pretrained_model= None):

        # ---create repos
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # ---gen, disc: true captions and images to feed the generator and generate fake captions
        n_examples = data['captions'].shape[0]
        n_iters_per_epoch = int(np.floor(float(n_examples) / batch_size))
        captions = data['captions']
        emotions = data['emotions']
        image_idxs = data['image_idxs']
        image_file_names = data['image_files_names']
        n_examples_val = val_data['captions'].shape[0]
        n_iters_val = int(np.floor(float(n_examples_val) / batch_size))
        captions_val = val_data['captions']
        emotions_val = val_data['emotions']
        image_idxs_val = val_data['image_idxs']
        image_file_names_val = val_data['image_files_names']

        # ---log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_iters_gen_loss = csv_logger(dir=log_path, file_name=timestamp + '_iters_gen_loss', first_row=['epoch', 'iteration', 'loss'])
        log_iters_disc_loss = csv_logger(dir=log_path, file_name=timestamp + '_iters_disc_loss', first_row=['epoch', 'iteration', 'loss'])
        log_iters_disc_accuracy = csv_logger(dir=log_path, file_name=timestamp + '_iters_disc_accuracy', first_row=['epoch', 'iteration', 'accuracy'])
        log_epoch_gen_loss = csv_logger(dir=log_path, file_name=timestamp + '_epoch_gen_loss', first_row=['epoch', 'loss'])
        log_epoch_gen_scores = csv_logger(dir=log_path, file_name=timestamp + '_epoch_gen_scores', first_row=['epoch',]+gen_scores)
        log_epoch_disc_loss = csv_logger(dir=log_path, file_name=timestamp + '_epoch_disc_loss', first_row=['epoch', 'loss'])
        log_epoch_disc_accuracy = csv_logger(dir=log_path, file_name=timestamp + '_epoch_disc_accuracy', first_row=['epoch', 'accuracy'])

        # ---define graph config
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # ---load pretrained model
        saver = tf.train.Saver(max_to_keep=40)
        if pretrained_model is not None:
            print("Training will start use a pretrained model")
            saver.restore(sess=self.sess, save_path=os.path.join(pretrained_model, 'model.ckpt'))

        # ---start training
        print('*' * 20 + "Start Training GAN" + '*' * 20)
        epoch_bar = tqdm(total=n_epochs)
        for e in range(n_epochs):
            self.curr_gen_loss = 0
            self.curr_disc_loss = 0
            # ---shuffle data
            rand_idxs = np.random.permutation(n_examples)
            captions = captions[rand_idxs]
            emotions = emotions[rand_idxs]
            image_idxs = image_idxs[rand_idxs]
            image_file_names = image_file_names[rand_idxs]
            # ---START training on one epoch
            iters_bar = tqdm(total=n_iters_per_epoch)
            for i in range(n_iters_per_epoch):
                # ---get one batch
                captions_batch = captions[i * batch_size:(i + 1) * batch_size]
                emotions_batch = emotions[i * batch_size:(i + 1) * batch_size]
                image_idxs_batch = image_idxs[i * batch_size:(i + 1) * batch_size]
                image_file_names_batch = image_file_names[i * batch_size:(i + 1) * batch_size]
                # ---extract features
                features_batch = self.generator.extract_features(self.sess, image_file_names_batch)
                # ---get reward from discriminator
                feed_dict = {self.generator.features: features_batch, self.generator.captions: captions_batch, self.generator.emotions: emotions_batch}
                fake_captions = self.sess.run(self.generator.generated_captions, feed_dict=feed_dict)
                rewards = rollout.get_reward(self.sess, fake_captions, self.generator.generated_captions, rollout_num, self.discriminator, features_batch, captions_batch)
                # ---train generator
                feed_dict = {self.generator.rewards: rewards, self.generator.features: features_batch,
                             self.generator.captions: captions_batch, self.generator.mode_learning: 2, self.generator.emotions: emotions_batch}
                _, l_gen = self.sess.run([self.generator.train_op, self.generator.loss], feed_dict=feed_dict)
                self.curr_gen_loss += l_gen
                # ---train discriminator
                fake_captions = self.sess.run(self.generator.generated_captions, feed_dict)
                real_captions = captions_batch[:, :self.generator.T]
                real_fake_captions = np.concatenate([real_captions, fake_captions], axis=0)
                fake_labels = [[1, 0] for _ in fake_captions]
                real_labels = [[0, 1] for _ in real_captions]
                real_fake_labels = np.concatenate([real_labels, fake_labels], axis=0)
                iters_disc_loss = []
                iters_disc_acc = []
                feed = {self.discriminator.input_x: real_fake_captions, self.discriminator.input_y: real_fake_labels,
                        self.discriminator.dropout_keep_prob: self.dis_dropout_keep_prob}
                for d_step in range(3):
                    _, loss, pred = self.sess.run([self.discriminator.train_op, self.discriminator.loss, self.discriminator.predictions], feed)
                    acc = np.mean(np.argmax(real_fake_labels, axis=1) == pred)
                    iters_disc_acc.append(acc)
                    iters_disc_loss.append(loss)
                    _ = self.sess.run(self.discriminator.params_clip, feed)

                # ---log
                if (i + 1) % log_every == 0:
                    pass
                iters_disc_acc = np.array(iters_disc_acc).mean()
                iters_disc_loss = np.array(iters_disc_loss).mean()
                self.curr_disc_loss += iters_disc_loss
                self.curr_disc_acc += iters_disc_acc
                iters_bar.update()
                iters_bar.set_description('Training: current disc loss/acc %f/%f%%' % (iters_disc_loss, iters_disc_acc * 100))
                log_iters_disc_loss.add_row([e + 1, i + 1, iters_disc_loss])
                log_iters_disc_accuracy.add_row([e + 1, i + 1, iters_disc_acc])

            self.curr_disc_loss /= n_iters_per_epoch
            self.curr_disc_acc /= n_iters_per_epoch
            log_epoch_disc_loss.add_row([e + 1, self.curr_disc_loss])
            log_epoch_disc_accuracy.add_row([e + 1, self.curr_disc_acc])

            # ---evaluate generated samples: compute scores score at every epoch on validation set
            if validation:
                # ---Init
                n_samples = val_features.shape[0]
                n_iters_val = int(np.ceil(float(n_samples) / batch_size))
                all_gen_cap = np.ndarray((val_features.shape[0], self.generator.T-4))
                # ---generate captions
                val_features[:, :, 2048:2052] = np.repeat(np.expand_dims(np.repeat([[0,1,0,1]], val_features.shape[1], axis=0), 0), val_features.shape[0], axis=0)
                for i in range(n_iters_val):
                    features_batch = val_features[i * batch_size:(i + 1) * batch_size]
                    feed_dict = {self.generator.features: features_batch,
                                 self.generator.whole_samples: captions_batch[:,4:self.generator.T],
                                 self.generator.nsample: 0, self.generator.mode_sampling: 1, self.generator.captions: captions_batch}
                    gen_cap = self.sess.run(self.generator.generated_captions, feed_dict=feed_dict)
                    all_gen_cap[i * batch_size:(i + 1) * batch_size] = gen_cap

                all_decoded = decode_captions(all_gen_cap, self.generator.idx_to_word)
                # ---store generated captions
                save_pickle(all_decoded, os.path.join(data_save_path, "val.candidate.captions.pkl"))
                # --compute scores
                scores = evaluate(data_path=data_save_path, split='val', get_scores=True)
                # ---log
                write_bleu(scores=scores, path=model_path, epoch=e, senti=[1])

                # ---generate captions
                val_features[:, :, 2048:2052] = np.repeat(np.expand_dims(np.repeat([[0, 0, 1, 2]], val_features.shape[1], axis=0), 0), val_features.shape[0], axis=0)
                for i in range(n_iters_val):
                    features_batch = val_features[i * batch_size:(i + 1) * batch_size]
                    feed_dict = {self.generator.features: features_batch, self.generator.whole_samples: captions_batch[:,4:self.generator.T],
                                 self.generator.nsample: 0, self.generator.mode_sampling: 1, self.generator.captions: captions_batch}
                    gen_cap = self.sess.run(self.generator.generated_captions, feed_dict=feed_dict)
                    all_gen_cap[i * batch_size:(i + 1) * batch_size] = gen_cap

                all_decoded = decode_captions(all_gen_cap, self.generator.idx_to_word)
                # ---store generated captions
                save_pickle(all_decoded, os.path.join(data_save_path, "val.candidate.captions.pkl"))
                # --compute scores
                scores = evaluate(data_path=data_save_path, split='val', get_scores=True)
                # ---log
                write_bleu(scores=scores, path=model_path, epoch=e, senti=[-1])

            # ---save model's parameters
            if (e + 1) % save_every == 0:
                saver.save(self.sess, os.path.join(model_path, "model.ckpt"))

            epoch_bar.update()
            epoch_bar.set_description('Training: disc previous - current epoch loss %f - %f / previous - current epoch acc %f%% - %f%%' % (self.prev_disc_loss, self.curr_disc_loss, self.prev_disc_acc * 100, self.curr_disc_acc * 100))
            self.prev_disc_loss = self.curr_disc_loss
            self.prev_disc_acc = self.curr_disc_acc