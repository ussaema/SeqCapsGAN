import tensorflow as tf
import numpy as np
import os

import numpy as np
from datetime import datetime
from tqdm import tqdm
from core.log import *
from core.utils import *

from core.capsLayer import CapsLayer


# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output




epsilon = 1e-9


class CapsNet(object):
    def __init__(self, is_training=True, height=28, width=28, channels=1, num_label=10):
        """
        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

        self.build_arch()

        # t_vars = tf.trainable_variables()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

        tf.logging.info('Seting up the main structure')






class Discriminator(object):

    def __init__(
            self, sess, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, pretrained_model= None, learning_rate=None, batch_size=16, model='capsnet'):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            if model == 'cnn':
                self.build_cnn(filter_sizes, num_filters, embedding_size, sequence_length, num_classes, l2_reg_lambda=l2_reg_lambda)
            elif model == 'capsnet':
                self.build_capsnet()

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        if learning_rate:
            d_optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = d_optimizer.minimize(self.loss)
            #grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
            #self.train_op = d_optimizer.apply_gradients(grads_and_vars)
            #self.params_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.params]

        # ---init
        self.prev_loss = -1
        self.prev_acc = -1

        self.sess = sess
        # ---load pretrained model
        self.saver = tf.train.Saver(max_to_keep=40)
        if pretrained_model is not None:
            print("Pretrained discriminator loaded")
            self.saver.restore(sess=self.sess, save_path=os.path.join(pretrained_model, 'model.ckpt'))
        initialize_uninitialized(self.sess)

    def build_capsnet(self, mask_with_y=True, stddev=0.01, iter_routing=3, m_plus=0.9, m_minus=0.1, lambda_val=0.5, regularization_scale=0.392):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return tensor with shape [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.embedded_chars_expanded, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')

        # Primary Capsules layer, return tensor with shape [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=8, batch_size=self.batch_size, stddev=stddev, iter_routing=iter_routing, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=4)

        # DigitCaps layer, return shape [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=2, batch_size=self.batch_size, stddev= stddev, iter_routing= iter_routing, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = softmax(self.v_length, axis=1)
            self.ypred_for_auc = tf.squeeze(self.softmax_v)
            # assert self.softmax_v.get_shape() == [cfg.batch_size, self.num_label, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            # assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.predictions = tf.reshape(self.argmax_idx, shape=(self.batch_size, ))

            # Method 1.
            if not mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for batch_size in range(self.batch_size):
                    v = self.caps2[batch_size][self.predictions[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [self.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.input_y, (-1, 2, 1)))
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2,
                                                             num_outputs=self.embedded_chars_expanded.shape[1].value * self.embedded_chars_expanded.shape[2].value * self.embedded_chars_expanded.shape[3].value,
                                                             activation_fn=tf.sigmoid)

        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - m_minus))
        assert max_l.get_shape() == [self.batch_size, 2, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(self.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.input_y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.embedded_chars_expanded, shape=(self.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.loss = self.margin_loss + regularization_scale * self.reconstruction_err

    def build_cnn(self, filter_sizes, num_filters, embedding_size, sequence_length, num_classes, l2_reg_lambda=0.0):
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(num_filters)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add highway
        with tf.name_scope("highway"):
            self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    def train(self, data, val_data, generator, n_epochs=10, batch_size=100, validation=False, dropout_keep_prob=1.0,  save_every= 1, log_every= 1, model_path='./model/discriminator/', pretrained_model=None,
              log_path='./log/discriminator/', iterations=1):

        sess = self.sess
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
        image_file_names_val= val_data['image_files_names']

        # ---log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_iters_loss = csv_logger(dir=log_path, file_name=timestamp + '_iters_loss', first_row=['epoch', 'iteration', 'loss'])
        log_iters_accuracy = csv_logger(dir=log_path, file_name=timestamp + '_iters_accuracy', first_row=['epoch', 'iteration', 'accuracy'])
        log_epoch_loss = csv_logger(dir=log_path, file_name=timestamp + '_epoch_loss', first_row=['epoch', 'loss'])
        log_epoch_accuracy = csv_logger(dir=log_path, file_name=timestamp + '_epoch_accuracy', first_row=['epoch', 'accuracy'])
        log_epoch_loss_val = csv_logger(dir=log_path, file_name=timestamp + '_epoch_loss_val', first_row=['epoch', 'loss'])
        log_epoch_accuracy_val = csv_logger(dir=log_path, file_name=timestamp + '_epoch_accuracy_val', first_row=['epoch', 'accuracy'])

        # ---START training of discriminator
        epoch_bar = tqdm(total=n_epochs)
        for e in range(n_epochs):
            self.curr_loss = 0
            self.curr_acc = 0
            # ---shuffle data
            rand_idxs = np.random.permutation(n_examples)
            captions = captions[rand_idxs]
            emotions = emotions[rand_idxs]
            image_idxs = image_idxs[rand_idxs]
            image_file_names = image_file_names[rand_idxs]
            # ---START training on one epoch
            iters_bar = tqdm(total=n_iters_per_epoch, position=0, leave=True)
            for i in range(n_iters_per_epoch):
                # ---get one batch
                captions_batch = captions[i * batch_size:(i + 1) * batch_size]
                emotions_batch = emotions[i * batch_size:(i + 1) * batch_size]
                image_idxs_batch = image_idxs[i * batch_size:(i + 1) * batch_size]
                image_file_names_batch = image_file_names[i * batch_size:(i + 1) * batch_size]
                # ---extract features
                features_batch = generator.extract_features(sess, image_file_names_batch)
                # ---train one step
                feed_dict = {generator.features: features_batch, generator.emotions: emotions_batch, generator.captions: captions_batch, generator.mode_learning: 1}
                accs = []
                losses = []
                for d_step in range(iterations):
                    # ---create a pair of fake and real captions
                    fake_captions = sess.run(generator.generated_captions, feed_dict)
                    real_captions = captions_batch[:, :generator.T]
                    real_fake_captions = np.concatenate([real_captions, fake_captions], axis=0)
                    fake_labels = [[1, 0] for _ in fake_captions]
                    real_labels = [[0, 1] for _ in real_captions]
                    real_fake_labels = np.concatenate([real_labels, fake_labels], axis=0)
                    idx = np.random.permutation(len(real_fake_labels))
                    real_fake_captions, real_fake_labels = real_fake_captions[idx], real_fake_labels[idx]

                    acc = 0
                    loss = 0
                    #first batch
                    feed = {self.input_x: real_fake_captions[:self.batch_size], self.input_y: real_fake_labels[:self.batch_size],
                            self.dropout_keep_prob: dropout_keep_prob}
                    _, loss_, pred = sess.run([self.train_op, self.loss, self.predictions], feed)
                    loss += loss_
                    acc += np.mean(np.argmax(real_fake_labels[:self.batch_size], axis=1) == pred)
                    #second batch
                    feed = {self.input_x: real_fake_captions[self.batch_size:], self.input_y: real_fake_labels[self.batch_size:],
                            self.dropout_keep_prob: dropout_keep_prob}
                    _, loss_, pred = sess.run([self.train_op, self.loss, self.predictions], feed)
                    loss += loss_
                    acc += np.mean(np.argmax(real_fake_labels[self.batch_size:], axis=1) == pred)

                    accs.append(acc/2)
                    losses.append(loss/2)

                # ---log
                if (i + 1) % log_every == 0:
                    pass
                losses = np.array(losses).mean()
                accs = np.array(accs).mean()
                self.curr_loss += losses
                self.curr_acc += accs
                iters_bar.update()
                iters_bar.set_description('Training: current loss/acc %f/%f%%' % (losses, accs*100))
                log_iters_loss.add_row([e + 1, i + 1, accs])
                log_iters_accuracy.add_row([e + 1, i + 1, losses])
            self.curr_loss /= n_iters_per_epoch
            self.curr_acc /= n_iters_per_epoch
            log_epoch_loss.add_row([e + 1, self.curr_loss])
            log_epoch_accuracy.add_row([e + 1, self.curr_acc])

            # ---print out validation loss and accuracies
            if validation:
                losses_val, accs_val = self.validate(sess, batch_size, captions_val, emotions_val, image_idxs_val, image_file_names_val, generator, dropout_keep_prob)
                log_epoch_loss_val.add_row([e + 1, losses_val])
                log_epoch_accuracy_val.add_row([e + 1, accs_val])

            # ---save model's parameters
            if (e + 1) % save_every == 0:
                self.saver.save(sess, os.path.join(model_path, "model.ckpt"))

            epoch_bar.update()
            epoch_bar.set_description('Training: previous - current epoch loss %f - %f / previous - current epoch acc %f%% - %f%%' % (self.prev_loss, self.curr_loss, self.prev_acc*100, self.curr_acc*100))
            self.prev_loss = self.curr_loss
            self.prev_acc = self.curr_acc

    def validate(self, sess, batch_size, captions_val, emotions_val, image_idxs_val, image_file_names_val, generator, dropout_keep_prob, verbose=1):
        n_iters_val = int(np.floor(float(captions_val.shape[0]) / batch_size))
        losses_val = []
        accs_val = []
        if verbose:
            iters_bar_val = tqdm(total=n_iters_val, position=1, leave=True)
        for i in range(n_iters_val):
            # ---get one batch
            captions_batch = captions_val[i * batch_size:(i + 1) * batch_size]
            emotions_batch = emotions_val[i * batch_size:(i + 1) * batch_size]
            image_idxs_batch = image_idxs_val[i * batch_size:(i + 1) * batch_size]
            image_file_names_batch = image_file_names_val[i * batch_size:(i + 1) * batch_size]
            # ---extract features
            features_batch = generator.extract_features(sess, image_file_names_batch)
            # ---train one step
            feed_dict = {generator.features: features_batch, generator.emotions: emotions_batch,
                         generator.captions: captions_batch, generator.mode_learning: 1}
            # ---create a pair of fake and real captions
            fake_captions = sess.run(generator.generated_captions, feed_dict)
            real_captions = captions_batch[:, :generator.T]
            real_fake_captions = np.concatenate([real_captions, fake_captions], axis=0)
            fake_labels = [[1, 0] for _ in fake_captions]
            real_labels = [[0, 1] for _ in real_captions]
            real_fake_labels = np.concatenate([real_labels, fake_labels], axis=0)
            feed = {self.input_x: real_fake_captions, self.input_y: real_fake_labels,
                    self.dropout_keep_prob: dropout_keep_prob}
            loss, pred = sess.run([self.loss, self.predictions], feed)
            acc = np.mean(np.argmax(real_fake_labels, axis=1) == pred)
            losses_val.append(loss)
            accs_val.append(acc)
            if verbose:
                iters_bar_val.update()
                iters_bar_val.set_description('Validation: loss/acc %f/%f%%' % (loss, acc * 100))
        losses_val = np.array(losses_val).mean()
        accs_val = np.array(accs_val).mean()
        if verbose:
            iters_bar_val.set_description('Validation: mean loss/acc %f/%f%%' % (losses_val, accs_val * 100))
        return losses_val, accs_val