import tensorflow as tf
import numpy as np
import os

import numpy as np
from datetime import datetime
from tqdm import tqdm
from core.log import *


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

class Discriminator(object):

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

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

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
        self.params_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.params]

        # ---init
        self.prev_loss = -1
        self.prev_acc = -1

    def train(self, sess, data, val_data, generator, n_epochs=10, batch_size=100, validation=False, dropout_keep_prob=1.0,  save_every= 1, log_every= 1, model_path='./model/discriminator/', pretrained_model=None,
              log_path='./log/discriminator/', iterations=1):
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

        # ---load pretrained model
        saver = tf.train.Saver(max_to_keep=40)
        if pretrained_model is not None:
            print("Training will start use a pretrained model")
            saver.restore(sess=self.sess, save_path=os.path.join(pretrained_model, 'model.ckpt'))

        # ---START training of discriminator
        print('*' * 20+ "Start Training"+ '*' * 20)
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
            iters_bar = tqdm(total=n_iters_per_epoch)
            for i in range(n_iters_per_epoch):
                # ---get one batch
                captions_batch = captions[i * batch_size:(i + 1) * batch_size]
                emotions_batch = emotions[i * batch_size:(i + 1) * batch_size]
                image_idxs_batch = image_idxs[i * batch_size:(i + 1) * batch_size]
                image_file_names_batch = image_file_names[i * batch_size:(i + 1) * batch_size]
                # ---extract features
                features_batch = generator.extract_features(self.sess, image_file_names_batch)
                # ---train one step
                feed_dict = {generator.features: features_batch, generator.emotions: emotions_batch, generator.captions: captions_batch, generator.mode_learning: 1}
                accs = []
                losses = []
                for d_step in range(iterations):
                    # ---create a pair of fake and real captions
                    fake_captions = self.sess.run(generator.generated_captions, feed_dict)
                    real_captions = captions_batch[:, :generator.T]
                    real_fake_captions = np.concatenate([real_captions, fake_captions], axis=0)
                    fake_labels = [[1, 0] for _ in fake_captions]
                    real_labels = [[0, 1] for _ in real_captions]
                    real_fake_labels = np.concatenate([real_labels, fake_labels], axis=0)
                    feed = {self.input_x: real_fake_captions, self.input_y: real_fake_labels,
                            self.dropout_keep_prob: dropout_keep_prob}
                    _, loss, pred = self.sess.run([self.train_op, self.loss, self.predictions], feed)
                    acc = np.mean(np.argmax(real_fake_labels, axis=1) == pred)
                    accs.append(acc)
                    losses.append(loss)

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
                log_iters_accuracy.add_row([e + 1, i + 1, accs])
            self.curr_loss /= n_iters_per_epoch
            self.curr_acc /= n_iters_per_epoch
            log_epoch_loss.add_row([e + 1, self.curr_loss])
            log_epoch_accuracy.add_row([e + 1, self.curr_acc])

            # ---print out validation loss and accuracies
            if validation:
                losses_val = []
                accs_val = []
                iters_bar_val = tqdm(total=n_iters_val)
                for i in range(n_iters_val):
                    # ---get one batch
                    captions_batch = captions_val[i * batch_size:(i + 1) * batch_size]
                    emotions_batch = emotions_val[i * batch_size:(i + 1) * batch_size]
                    image_idxs_batch = image_idxs_val[i * batch_size:(i + 1) * batch_size]
                    image_file_names_batch = image_file_names_val[i * batch_size:(i + 1) * batch_size]
                    # ---extract features
                    features_batch = generator.extract_features(self.sess, image_file_names_batch)
                    # ---train one step
                    feed_dict = {generator.features: features_batch, generator.emotions: emotions_batch,
                                 generator.captions: captions_batch, generator.mode_learning: 1}
                    # ---create a pair of fake and real captions
                    fake_captions = self.sess.run(generator.generated_captions, feed_dict)
                    real_captions = captions_batch[:, :generator.T]
                    real_fake_captions = np.concatenate([real_captions, fake_captions], axis=0)
                    fake_labels = [[1, 0] for _ in fake_captions]
                    real_labels = [[0, 1] for _ in real_captions]
                    real_fake_labels = np.concatenate([real_labels, fake_labels], axis=0)
                    feed = {self.input_x: real_fake_captions, self.input_y: real_fake_labels,
                            self.dropout_keep_prob: dropout_keep_prob}
                    loss, pred = self.sess.run([self.loss, self.predictions], feed)
                    acc = np.mean(np.argmax(real_fake_labels, axis=1) == pred)
                    losses_val.append(loss)
                    accs_val.append(acc)
                    iters_bar_val.update()
                    iters_bar_val.set_description('Validation: loss/acc %f/%f%%' % (loss, acc * 100))
                losses_val = np.array(losses_val).mean()
                accs_val = np.array(accs_val).mean()
                log_epoch_loss_val.add_row([e + 1, losses_val])
                log_epoch_accuracy_val.add_row([e + 1, accs_val])
                iters_bar_val.set_description('Validation: mean loss/acc %f/%f%%' % (losses_val, accs_val* 100))

            # ---save model's parameters
            if (e + 1) % save_every == 0:
                saver.save(self.sess, os.path.join(model_path, "model.ckpt"))

            epoch_bar.update()
            epoch_bar.set_description('Training: previous - current epoch loss %f - %f / previous - current epoch acc %f%% - %f%%' % (self.prev_loss, self.curr_loss, self.prev_acc*100, self.curr_acc*100))
            self.prev_loss = self.curr_loss
            self.prev_acc = self.curr_acc