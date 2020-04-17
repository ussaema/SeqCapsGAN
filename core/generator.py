# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================



import tensorflow as tf
from core.vggnet import Vgg19
import torch
from core.resnet import *
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import pickle as pickle
from scipy import ndimage
from .utils import *
from .bleu import evaluate
from matplotlib.pyplot import imread
import torch
from core.log import *
from tqdm import tqdm
import logging, os
from datetime import datetime
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(object):
    def __init__(self, word_to_idx, dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, emo2out=True, alpha_c=0.0, selector=True, dropout=True,
                 update_rule='adam', learning_rate=0.001,
                 vgg_model_path='./data/imagenet-vgg-verydeep-19.mat', features_extractor = 'vgg'):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.emo2out = emo2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)

        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step + 1
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        self.features_extractor = features_extractor
        if features_extractor == 'vgg':
            self.vggnet = Vgg19(vgg_model_path)
            dim_feature = [196, 512]
        elif features_extractor == 'resnet':
            self.resnet152 = ResNetFeatureExtractor(M.resnet152(pretrained=True).to(device), feat_layer="res5c")
            dim_feature = [49, 2048]

        self.L = dim_feature[0]
        self.D = dim_feature[1]

        # Placeholders
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.emotions = tf.placeholder(tf.float32, [None, 3])
        self.rewards = tf.placeholder(tf.float32, shape=[None, self.T])  # get from rollout policy and discriminator
        self.mode_learning = tf.placeholder(tf.int32)

        # Build graphs for training model and sampling captions
        with tf.variable_scope(tf.get_variable_scope()):
            self.loss = self.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, self.generated_captions = self.build_sampler()

        # ---set an optimizer by update rule
        if update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        # ---train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = self.optimizer(learning_rate=learning_rate)
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.grads_and_vars = list(zip(grads, tf.trainable_variables()))
            self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars)

        # ---init
        self.prev_loss = -1

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, emotion, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x

            if self.emo2out:
                w_emo2out = tf.get_variable('w_e', [3, self.M], initializer=self.weight_initializer)
                b_emo2out = tf.get_variable('b_e', [self.M], initializer=self.const_initializer)
                h_logits += tf.matmul(emotion, w_emo2out) + b_emo2out

            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):

        mode = self.mode_learning
        features = self.features
        captions = self.captions
        emotions = self.emotions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # ---feature extractor
        self.vggnet.build()
        features = self._batch_norm(features, mode='train', name='conv_features')
        # ---embedding layer
        x = self._word_embedding(inputs=captions_in)
        # ---feature projector
        features_proj = self._project_features(features=features)
        # --lstm
        c, h = self._get_initial_lstm(features=features)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        loss = 0.0
        alpha_list = []
        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
            context = tf.nn.dropout(context, 0.5) #TODO delete

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], context, emotions],1), state=[c, h])

            logits = self._decode_lstm(x[:,t,:], h, context, emotions, dropout=self.dropout, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t],logits=logits)*mask[:, t] )

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)
            loss += alpha_reg

        loss= tf.cond(mode<2, lambda: loss / tf.to_float(batch_size),
                      lambda: (loss / tf.to_float(batch_size)) + 0.01*tf.reduce_sum(tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(self.captions[:,:self.T], [-1])), self.V, 1.0, 0.0), 1) * tf.reshape(self.rewards, [-1]))/tf.to_float(batch_size) )
        return loss

    def build_sampler(self, seq_len = None):
        if seq_len is None:
            seq_len = self.T
        features = self.features
        emotions = self.emotions

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(seq_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context, emotions], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, emotions, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)

        return alphas, betas, sampled_captions
    def extract_features(self, sess, image_file_names_batch):
        images = []
        for image in image_file_names_batch:
            imm = imread(image)
            if len(imm.shape) == 3:
                images.append(imm)
            else:
                im = np.array([imm, ] * 3)
                images.append(im.reshape((im.shape[1], im.shape[2], im.shape[0])))
        image_batch = np.array(images).astype(np.float32)
        if self.features_extractor == 'vgg':
            features_batch = sess.run(self.vggnet.features, feed_dict={self.vggnet.images: image_batch})
        elif self.features_extractor == 'resnet':
            image_batch = torch.tensor(image_batch).view(-1, image_batch.shape[3], image_batch.shape[1],
                                                         image_batch.shape[2]).to(device)
            features_batch = self.resnet152(image_batch)
            features_batch = features_batch.cpu().detach().numpy().reshape(-1, 49, 2048)
        return features_batch

    def train(self, sess, data, val_data, n_epochs= 10, batch_size= 100,
              validation= False, scores=['Bleu_1','Bleu_2','Bleu_3','Bleu_4', 'ROUGE_L', 'CIDEr'], save_every= 1, log_every=1, log_path= './log/generator/', model_path= './model/generator/',
              pretrained_model= None, reward=None):
        self.sess = sess
        # ---create repos
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # ---get train data
        n_examples = data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / batch_size))
        captions = data['captions']
        emotions = data['emotions']
        references = data['references']
        image_idxs = data['image_idxs']
        image_file_names = data['image_files_names']

        # ---get validation data
        n_examples_val = val_data['file_names'].shape[0]
        n_iters_per_epoch_val = int(np.ceil(float(n_examples_val) / batch_size))
        references_val = val_data['references']
        file_names_val = val_data['file_names']
        references_emotions_val = val_data['references_emotions']

        # ---get reward
        if not reward:
            rewards = np.zeros((batch_size, self.T), dtype=np.float32)

        # ---log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_generated_captions = csv_logger(dir=log_path, file_name=timestamp+'_captions', first_row=['epoch', 'image', 'generated', 'gt1', 'gt2', 'gt3', 'gt4', 'gt5'])
        log_iters_loss = csv_logger(dir=log_path, file_name=timestamp+'_iters_loss', first_row=['epoch', 'iteration', 'loss'])
        log_epoch_loss = csv_logger(dir=log_path, file_name=timestamp+'_epoch_loss', first_row=['epoch', 'loss'])
        log_epoch_scores_val = csv_logger(dir=log_path, file_name=timestamp+'_epoch_scores_val', first_row=['epoch',]+scores)

        # ---summary op
        """"summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        tf.summary.scalar('batch_loss', self.loss)
        if log_all:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            for grad, var in self.grads_and_vars:
                try:
                    tf.summary.histogram(var.op.name + '/gradient', grad)
                except:
                    pass
        self.summary_op = tf.summary.merge_all()"""

        # ---define graph config
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # ---load pretrained model
        saver = tf.train.Saver(max_to_keep=40)
        if pretrained_model is not None:
            print("Training will start use a pretrained model")
            saver.restore(sess=self.sess, save_path= os.path.join(pretrained_model, 'model.ckpt'))

        # ---start training
        print('*' * 20+ "Start Generator Training"+ '*' * 20)
        epoch_bar = tqdm(total=n_epochs)
        for e in range(n_epochs):
            self.curr_loss = 0
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
                features_batch = self.extract_features(self.sess, image_file_names_batch)
                # ---train one step
                feed_dict = {self.features: features_batch, self.emotions: emotions_batch, self.captions: captions_batch, self.rewards: rewards, self.mode_learning: 1}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict)
                self.curr_loss += l
                # ---write summary for tensorboard visualization
                """if i % 10 == 0:
                    summary = self.sess.run(self.summary_op, feed_dict)
                    summary_writer.add_summary(summary, e * n_iters_per_epoch + i)"""
                # --- log
                if (i + 1) % log_every == 0:
                    ground_truths = captions[image_idxs == image_idxs_batch[0]]
                    decoded = decode_captions(ground_truths, self.idx_to_word)
                    gt_list = []
                    for j, gt in enumerate(decoded):
                        gt_list.append(gt)
                    gen_caps = self.sess.run(self.generated_captions, feed_dict)
                    decoded = decode_captions(gen_caps, self.idx_to_word)
                    log_generated_captions.add_row([e+1, image_file_names_batch[0], decoded[0],]+gt_list)
                iters_bar.update()
                iters_bar.set_description('Training: current loss %d' % (l))
                log_iters_loss.add_row([e + 1, i + 1, l])
            self.curr_loss /= n_iters_per_epoch
            log_epoch_loss.add_row([e+1, self.curr_loss])

            # ---print out BLEU scores and file write
            if validation:
                scores = self.validate(batch_size, file_names_val, references_emotions_val, references_val, scores)
                log_epoch_scores_val.add_row([e + 1, ] + list(scores.values()))

            # ---save model's parameters
            if (e + 1) % save_every == 0:
                saver.save(self.sess, os.path.join(model_path, "model.ckpt"))

            epoch_bar.update()
            epoch_bar.set_description('Training: previous - current epoch loss %f - %f'%(self.prev_loss, self.curr_loss))
            self.prev_loss = self.curr_loss

    def validate(self, batch_size, file_names_val, references_emotions_val, references_val, scores, verbose = 1):
        n_iters_per_epoch_val = int(np.ceil(float(file_names_val.shape[0]) / batch_size))
        all_gen_cap = []
        if verbose:
            val_iters_bar = tqdm(total=n_iters_per_epoch_val)
        for i in range(n_iters_per_epoch_val):
            image_file_names_batch_val = file_names_val[i * batch_size:(i + 1) * batch_size]
            emotions_batch_val = references_emotions_val[i * batch_size:(i + 1) * batch_size]
            features_batch = self.extract_features(self.sess, image_file_names_batch_val)
            feed_dict = {self.features: features_batch, self.emotions: emotions_batch_val}
            gen_cap = self.sess.run(self.generated_captions, feed_dict=feed_dict)
            all_gen_cap.append(gen_cap)
            if verbose:
                val_iters_bar.update()

        all_decoded = decode_captions(np.vstack(all_gen_cap), self.idx_to_word)
        scores = evaluate(all_decoded, references_val, scores)
        if verbose:
            val_iters_bar.set_description(
                'Validation: Bleu_1: %f' % scores['Bleu_1'] + ' Bleu_2: %f' % scores['Bleu_2'] + ' Bleu_3: %f' % scores[
                    'Bleu_3'] + ' Bleu_4: %f' % scores['Bleu_4'] + ' ROUGE_L: %f' % scores['ROUGE_L'] + ' CIDEr: %f' %
                scores['CIDEr'])
        return scores

    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.idx_to_word)

            if attention_visualization:
                for n in range(10):
                    print("Sampled Caption: %s" %decoded[n])

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))