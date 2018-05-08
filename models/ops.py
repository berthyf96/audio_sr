from __future__ import division
import tensorflow as tf
import os
import sys
import numpy as np
import librosa
import fnmatch

# -------------------------------
# MODEL OPS
# -------------------------------
def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(
        learning_rate=learning_rate, epsilon=1e-4)

def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)

def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(
        learning_rate=learning_rate, momentum=momentum, epsilon=1e-5)

optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print("Storing checkpoint to {} ...".format(logdir))
    sys.stdout.flush()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    return

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir))
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("Checkpoint found: {}".format(
            ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1]
                          .split('-')[-1])
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored model from global step {}".format(
            global_step))
        return global_step
    else:
        print("No checkpoint found")
        return None
    return None

def scalar_summary(name, x):
    try:
        summ = tf.summary.scalar(name, x)
    except AttributeError:
        summ = tf.scalar_summary(name, x)
    return summ

# -------------------------------
# I/O OPS
# -------------------------------
def find_files(directory, pattern='*.wav'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def get_test_batches(files, batch_size, sample_rate):
    # Grab batch_size number of audio files
    nb_audio_batch = []
    wb_audio_batch = []
    for i in range(batch_size):
        nb_filename = np.random.choice(files)
        wb_filename = nb_filename.replace('nb', 'wb')
        nb_audio, _ = librosa.load(
            nb_filename, sr=sample_rate, mono=True)
        wb_audio, _ = librosa.load(
            wb_filename, sr=sample_rate, mono=True)
        nb_audio = nb_audio.reshape(-1, 1)
        wb_audio = wb_audio.reshape(-1, 1)
        nb_audio_batch.append(nb_audio)
        wb_audio_batch.append(wb_audio)
    nb_audio_batch = np.asarray(nb_audio_batch)
    wb_audio_batch = np.asarray(wb_audio_batch)
    nb_audio_batch = nb_audio_batch.reshape(batch_size, -1, 1)
    wb_audio_batch = wb_audio_batch.reshape(batch_size, -1, 1)
    return nb_audio_batch, wb_audio_batch

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))
    return

# -------------------------------
# MATH OPS
# -------------------------------
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def average_gradients(tower_grads):
    '''Returns: List of pairs of (gradient, variable), where the gradient
    has been averaged across all towers'''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# -------------------------------
# SIGNAL PROCESSING OPS
# -------------------------------
def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log(1 + mu * safe_audio_abs) / tf.log(1. + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1.0) / 2 * mu + 0.5, tf.int32)
    
def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float64)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1.0 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

def log_mel_spectrograms(signals, sample_rate):
    fft_length = 512
    signals = tf.cast(signals, dtype=tf.float32)
    stfts = tf.contrib.signal.stft(
        signals, frame_length=512, frame_step=64, 
        fft_length=fft_length)
    magnitude_spectrograms = tf.abs(stfts)
    # Warp the linear-scale, magnitude spectrograms into mel-scale.
    num_spectrogram_bins = fft_length // 2 + 1
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = \
        80.0, 7600.0, 64
    linear_to_mel_weight_matrix = \
        tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, 
            lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
    return log10(mel_spectrograms)

# -------------------------------
# NEURAL NET OPS
# -------------------------------
def downconv(x, output_dim, kwidth=5, pool=2, init=None, uniform=False,
             bias_init=None, name='downconv'):
    """ Downsampled convolution 1d """
    x2d = tf.expand_dims(x, 2)
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable(
            'W', [kwidth, 1, x.get_shape()[-1], output_dim], 
            initializer=w_init)
        conv = tf.nn.conv2d(
            x2d, W, strides=[1, pool, 1, 1], padding='SAME')
        if bias_init is not None:
            b = tf.get_variable(
                'b', [output_dim], initializer=bias_init)
            conv = tf.reshape(
                tf.nn.bias_add(conv, b), conv.get_shape())
        else:
            conv = tf.reshape(conv, conv.get_shape())
        conv = tf.reshape(conv, conv.get_shape().as_list()[:2] +
                          [conv.get_shape().as_list()[-1]])
        return conv
    
def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)

def prelu(x, name='prelu', ref=False):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # make one alpha per feature
        alpha = tf.get_variable(
            'alpha', in_shape[-1], 
            initializer=tf.constant_initializer(0.), 
            dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
        if ref:
            # return ref to alpha vector
            return pos + neg, alpha
        else:
            return pos + neg

def conv1d(x, kwidth=5, num_kernels=1, 
           init=None, uniform=False, bias_init=None,
           name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        # filter shape: [kwidth, in_channels, num_kernels]
        W = tf.get_variable('W', [kwidth, in_channels, num_kernels],
                            initializer=w_init
                            )
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        if bias_init is not None:
            b = tf.get_variable(
                'b', [num_kernels], 
                initializer=tf.constant_initializer(bias_init))
            conv = conv + b
        return conv

