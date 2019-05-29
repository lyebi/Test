import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
from attention import attention
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        # Use dropout here to help prevent overfitting?
        #return tf.nn.dropout(tf.nn.conv2d(x, w, stride_shape, pad) + b, keep_prob=0.7, name='dropout_%s' % name)
        # Turn out it is better without dropout
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b



def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, meta_ac_space):


        with tf.variable_scope('conv'):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

            x = tf.nn.relu( conv2d(x, 16, "l1", [8, 8], [4, 4]) )
            self.new_conv_feature = tf.reduce_mean(x,axis=[1,2])


            x = tf.nn.relu( conv2d(x, 32, "l2", [4, 4], [2, 2]) )
            # x is [?, 11, 11, 32]

            self.conv_feature = tf.reduce_mean(x, axis=[1,2])
            self.conv_feature2=x  #None ,11,11,32

            x = tf.nn.relu(linear(flatten(x), 256, "hidden",  normalized_columns_initializer(1.0)))

            self.prev_action = prev_action = tf.placeholder(tf.float32, [None, ac_space], "prev_a")
            self.prev_reward = prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_r")

            # concat previous action and reward
            x = tf.concat([x, prev_action], axis=1)
            x = tf.concat([x, prev_reward], axis=1)

            self.meta_action = meta_action = tf.placeholder(tf.float32, [None, meta_ac_space], "meta_action")

            # concat
            x = tf.concat([x, meta_action], axis=1)
            # attention_mask = attention(x, 332)
            # print(attention_mask.get_shape())
            # x = tf.multiply(x, attention_mask) + x

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

        with tf.variable_scope('lstm'):
            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]
            print('a:',tf.shape(self.x))
            print('b:',step_size)


            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)


            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            print('lstm_state:', lstm_outputs.get_shape())


            lstm_c, lstm_h = lstm_state




            x = tf.reshape(lstm_outputs, [-1, size])
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = categorical_sample(self.logits, ac_space)[0, :]

        # Note: need to be on scope of the class
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h, prev_a, prev_r, meta_a):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c,
                        self.state_in[1]: h, self.prev_action: [prev_a],
                        self.prev_reward: [prev_r], self.meta_action: [meta_a] })

    def value(self, ob, c, h, prev_a, prev_r, meta_a):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c,
                        self.state_in[1]: h, self.prev_action: [prev_a],
                        self.prev_reward: [prev_r], self.meta_action: [meta_a]})[0]

    def get_conv_feature(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.conv_feature], {self.x: [ob]})
    def get_new_conv_feature(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.new_conv_feature], {self.x: [ob]})
    def get_conv_feature2(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.conv_feature2], {self.x: [ob]})


class MetaPolicy(object):
    def __init__(self, ob_space, ac_space = 37):

        with tf.variable_scope('conv1'):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

            x = tf.nn.relu( conv2d(x, 16, "l1", [8, 8], [4, 4]) )
            x = tf.nn.relu( conv2d(x, 32, "l2", [4, 4], [2, 2]) )
            x = tf.nn.relu(linear(flatten(x), 256, "hidden",  normalized_columns_initializer(1.0)))

            self.prev_action = prev_action = tf.placeholder(tf.float32, [None, ac_space], "prev_a")
            self.prev_reward = prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_r")

            # concat previous action and reward
            x = tf.concat([x, prev_action], axis=1)
            x = tf.concat([x, prev_reward], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

        with tf.variable_scope('meta_lstm1'):
            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

            # lstm output
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            # try logits with 36 actions + 1 no patch action
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r] })

    def value(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r]})[0]



class MetaPolicy2(object):
    def __init__(self, ob_space, ac_space = 37):

        with tf.variable_scope('conv2'):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

            x = tf.nn.relu( conv2d(x, 16, "l1", [8, 8], [4, 4]) )
            x = tf.nn.relu( conv2d(x, 32, "l2", [4, 4], [2, 2]) )
            x = tf.nn.relu(linear(flatten(x), 256, "hidden",  normalized_columns_initializer(1.0)))

            self.prev_action = prev_action = tf.placeholder(tf.float32, [None, ac_space], "prev_a")
            self.prev_reward = prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_r")

            # concat previous action and reward
            x = tf.concat([x, prev_action], axis=1)
            x = tf.concat([x, prev_reward], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

        with tf.variable_scope('meta_lstm2'):
            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

            # lstm output
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            # try logits with 36 actions + 1 no patch action
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r] })

    def value(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r]})[0]



class MetaPolicy3(object):
    def __init__(self, ob_space, ac_space = 37):

        with tf.variable_scope('conv3'):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

            x = tf.nn.relu( conv2d(x, 16, "l1", [8, 8], [4, 4]) )
            x = tf.nn.relu( conv2d(x, 32, "l2", [4, 4], [2, 2]) )
            x = tf.nn.relu(linear(flatten(x), 256, "hidden",  normalized_columns_initializer(1.0)))

            self.prev_action = prev_action = tf.placeholder(tf.float32, [None, ac_space], "prev_a")
            self.prev_reward = prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_r")

            # concat previous action and reward
            x = tf.concat([x, prev_action], axis=1)
            x = tf.concat([x, prev_reward], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

        with tf.variable_scope('meta_lstm3'):
            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

            # lstm output
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            # try logits with 36 actions + 1 no patch action
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r] })

    def value(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r]})[0]


class MetaPolicy4(object):
    def __init__(self, ob_space, ac_space = 37):

        with tf.variable_scope('conv4'):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

            x = tf.nn.relu( conv2d(x, 16, "l1", [8, 8], [4, 4]) )
            x = tf.nn.relu( conv2d(x, 32, "l2", [4, 4], [2, 2]) )
            x = tf.nn.relu(linear(flatten(x), 256, "hidden",  normalized_columns_initializer(1.0)))

            self.prev_action = prev_action = tf.placeholder(tf.float32, [None, ac_space], "prev_a")
            self.prev_reward = prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_r")

            # concat previous action and reward
            x = tf.concat([x, prev_action], axis=1)
            x = tf.concat([x, prev_reward], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

        with tf.variable_scope('meta_lstm4'):
            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

            # lstm output
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            # try logits with 36 actions + 1 no patch action
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            # self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run([self.logits, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r] })

    def value(self, ob, c, h, prev_a, prev_r):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.prev_action: [prev_a], self.prev_reward: [prev_r]})[0]