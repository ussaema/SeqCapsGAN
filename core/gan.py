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
        self.prev_disc_acc = -1
        self.prev_disc_loss = -1

    def get_reward(self, sess, features_batch, emotions_batch, captions_batch, rollout_num):
        rewards = []
        for i in range(rollout_num):
            # given_num between 1 to sequence_length - 1 for a part completed sentence
            feed_dict = {self.generator.features: features_batch, self.generator.emotions: emotions_batch}
            generated_captions = sess.run(self.generator.generated_captions, feed_dict)
            for seq_length in range(1, self.generator.T):
                feed_dict = {self.discriminator.input_x: np.column_stack((generated_captions[:,:seq_length], np.zeros((generated_captions.shape[0], self.generator.T-seq_length), dtype=np.int64))) , self.discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(self.discriminator.ypred_for_auc, feed_dict)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[seq_length - 1] += ypred

            # the last token reward
            feed_dict = {self.discriminator.input_x: captions_batch[:,:self.generator.T], self.discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(self.discriminator.ypred_for_auc, feed_dict)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                # completed sentence reward
                rewards[self.generator.T - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def train(self, sess, data, val_data, n_epochs=10, batch_size=100, rollout_num=10, validation=False, gen_scores=['Bleu_1','Bleu_2','Bleu_3','Bleu_4', 'ROUGE_L', 'CIDEr'], log_every=100, save_every=1, log_path='./log/', model_path='./model/', pretrained_model= None):
        self.sess = sess
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
        captions_val = val_data['captions']
        emotions_val = val_data['emotions']
        image_idxs_val = val_data['image_idxs']
        image_file_names_val = val_data['image_files_names']
        file_names_val = val_data['file_names']
        references_val = val_data['references']
        references_emotions_val = val_data['references_emotions']

        # ---log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_iters_gen_loss = csv_logger(dir=log_path, file_name=timestamp + '_iters_gen_loss', first_row=['epoch', 'iteration', 'loss'])
        log_iters_disc_loss = csv_logger(dir=log_path, file_name=timestamp + '_iters_disc_loss', first_row=['epoch', 'iteration', 'loss'])
        log_iters_disc_accuracy = csv_logger(dir=log_path, file_name=timestamp + '_iters_disc_accuracy', first_row=['epoch', 'iteration', 'accuracy'])
        log_epoch_gen_loss = csv_logger(dir=log_path, file_name=timestamp + '_epoch_gen_loss', first_row=['epoch', 'loss'])
        log_epoch_disc_loss = csv_logger(dir=log_path, file_name=timestamp + '_epoch_disc_loss', first_row=['epoch', 'loss'])
        log_epoch_disc_accuracy = csv_logger(dir=log_path, file_name=timestamp + '_epoch_disc_accuracy', first_row=['epoch', 'accuracy'])
        log_epoch_gen_scores_val = csv_logger(dir=log_path, file_name=timestamp + '_epoch_gen_scores_val', first_row=['epoch', ] + gen_scores)
        log_epoch_disc_loss_val = csv_logger(dir=log_path, file_name=timestamp + '_epoch_disc_loss_val', first_row=['epoch', 'loss'])
        log_epoch_disc_accuracy_val = csv_logger(dir=log_path, file_name=timestamp + '_epoch_disc_accuracy_val', first_row=['epoch', 'accuracy'])

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
            self.curr_disc_acc = 0
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
                rewards = self.get_reward(self.sess, features_batch, emotions_batch, captions_batch, rollout_num)
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
                iters_bar.set_description('Training: current disc loss/acc %f/%f%% / gen loss %f' % (iters_disc_loss, iters_disc_acc * 100, l_gen))
                log_iters_disc_loss.add_row([e + 1, i + 1, iters_disc_loss])
                log_iters_disc_accuracy.add_row([e + 1, i + 1, iters_disc_acc])

            self.curr_disc_loss /= n_iters_per_epoch
            self.curr_disc_acc /= n_iters_per_epoch
            log_epoch_disc_loss.add_row([e + 1, self.curr_disc_loss])
            log_epoch_disc_accuracy.add_row([e + 1, self.curr_disc_acc])

            # ---evaluate generated samples: compute scores score at every epoch on validation set
            if validation:
                losses_val, accs_val = self.discriminator.validate(batch_size, captions_val, emotions_val,
                                                     image_idxs_val, image_file_names_val, self.generator, self.dis_dropout_keep_prob, verbose=0)
                log_epoch_disc_loss_val.add_row([e + 1, losses_val])
                log_epoch_disc_accuracy_val.add_row([e + 1, accs_val])
                scores = self.generator.validate(batch_size, file_names_val, references_emotions_val, references_val, gen_scores, verbose=0) #TODO same scores are appended!!!
                log_epoch_gen_scores_val.add_row([e + 1, ] + list(scores.values()))

            # ---save model's parameters
            if (e + 1) % save_every == 0:
                saver.save(self.sess, os.path.join(model_path, "model.ckpt"))

            epoch_bar.update()
            epoch_bar.set_description('Training: disc previous - current epoch loss %f - %f / previous - current epoch acc %f%% - %f%% / gen previous - current epoch loss %f - %f' % (self.prev_disc_loss, self.curr_disc_loss, self.prev_disc_acc * 100, self.curr_disc_acc * 100, self.prev_gen_loss, self.curr_gen_loss))
            self.prev_disc_loss = self.curr_disc_loss
            self.prev_disc_acc = self.curr_disc_acc
            self.prev_gen_loss = self.curr_gen_loss