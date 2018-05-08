import tensorflow as tf
import librosa
import numpy as np
import argparse
from models import HRNN, HRNN_GAN, Discriminator
from models import write_wav, log_mel_spectrograms
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--big_frame_size', type=int, default=8)
    parser.add_argument('--frame_size', type=int, default=2)
    parser.add_argument('--q_levels', type=int, default=256)
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--n_rnn', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=520)
    parser.add_argument('--emb_size', type=int, default=256)
    parser.add_argument('--spec_loss_weight', type=float, required=False)
    parser.add_argument('--l1_reg_strength', type=float, default=0.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--step', type=int, default=700)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--inp_file', type=str, required=True)
    return parser.parse_args()

def create_hrnn(args):
    net = HRNN(args)
    return net

def create_gan(args):
    net = HRNN_GAN(batch_size=args.batch_size,
                   big_frame_size=args.big_frame_size,
                   frame_size=args.frame_size,
                   q_levels=args.q_levels,
                   rnn_type=args.rnn_type,
                   dim=args.dim,
                   n_rnn=args.n_rnn,
                   seq_len=args.seq_len,
                   emb_size=args.emb_size)
    return net

def load_step(saver, sess, logdir, step):
    print("Trying to restore saved checkpoints from {} ...".format(logdir))
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1]
                          .split('-')[-1])
        model_path = '{}/model.ckpt-{}'.format(logdir, step)
        saver.restore(sess, model_path)
        print("Restored model from global step {}".format(step))
        return global_step
    else:
        print("No checkpoint found")
        return None
    return None

def load_audio(nb_file, wb_file):
    nb_audio, _ = librosa.load(nb_file, sr=16000, mono=True)
    wb_audio, _ = librosa.load(wb_file, sr=16000, mono=True)
    return nb_audio, wb_audio

def crossfade(s1, s2, overlap):
    s1_stop = len(s1) - overlap
    res = np.zeros(len(s1) + len(s2) - overlap)
    res[:s1_stop] = s1[:s1_stop]
    for i in range(overlap):
        alpha = float(i) / (overlap - 1)
        res[s1_stop + i] = (alpha * s2[i]) + ((1 - alpha) * s1[s1_stop + i])
    res[s1_stop + overlap: ] = s2[overlap: ]
    return res

def l1_loss(pred, target):
    return np.mean(np.absolute(pred - target))

def log_spectral_distance(pred, target, sess):
    pred_spectrogram_tensor = log_mel_spectrograms(pred, 16000)
    target_spectrogram_tensor = log_mel_spectrograms(target, 16000)
    pred_spectrogram, target_spectrogram = sess.run(
        [pred_spectrogram_tensor, target_spectrogram_tensor])
    squared_distance = np.square(pred_spectrogram - target_spectrogram)
    return np.mean(squared_distance)

def evaluate(args):
    if args.method == 'baseline' or args.method == 'spec':
        args.spec_loss_weight = 0.0
        net = create_hrnn(args)
    elif args.method == 'gan':
        net = create_gan(args)
    else:
        raise ValueError('Please specify a method (baseline, spec, or gan).')
    
    # input placeholders
    nb_input_batch = tf.Variable(
        tf.zeros([net.batch_size, net.seq_len, 1]),
        trainable=False,
        dtype=tf.float32)
    wb_input_batch = tf.Variable(
        tf.zeros([net.batch_size, net.seq_len, 1]),
        trainable=False,
        dtype=tf.float32)
    
    # initial lstm states
    train_big_frame_state = net.big_cell.zero_state(
        net.batch_size, tf.float32)
    train_frame_state = net.cell.zero_state(net.batch_size, tf.float32)
    final_big_frame_state_spec = net.big_cell.zero_state(
        net.batch_size, tf.float32)
    final_frame_state_spec = net.cell.zero_state(net.batch_size, tf.float32)
    
    # output variables
    if args.method == 'baseline' or args.method == 'spec':
        loss, prediction, final_big_frame_state, final_frame_state = \
            net.forward(
                nb_input_batch, wb_input_batch, train_big_frame_state, 
                train_frame_state, inference_only=True)
    else:
        loss, final_big_frame_state, final_frame_state, _, prediction = \
            net.loss_SampleRNN(
                nb_input_batch, wb_input_batch, train_big_frame_state, 
                train_frame_state)
    
    # configure session
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    
    # load saved model
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    logdir = args.logdir
    load_step(saver, sess, logdir, args.step)
    
    test_nb_file = args.inp_file
    test_wb_file = test_nb_file.replace('nb', 'wb')
    nb_audio, wb_audio = load_audio(test_nb_file, test_wb_file)
    result = np.zeros(len(nb_audio) - 8)
    nb_audio = nb_audio.reshape(-1, 1)
    wb_audio = wb_audio.reshape(-1, 1)
    output_list = [loss, prediction, final_big_frame_state, final_frame_state]
    sample_size = len(nb_audio)
    seq_len = 520
    stride = 256
    overlap = 256
    print('Running model...')
    for i in range(0, sample_size, stride):
        if (i + seq_len) >= len(nb_audio): break
        inp_dict = {}
        inp_dict[nb_input_batch] = [nb_audio[i:i + seq_len]]
        inp_dict[wb_input_batch] = [wb_audio[i:i + seq_len]]
        inp_dict[train_big_frame_state] = sess.run(net.big_initial_state)
        inp_dict[train_frame_state] = sess.run(net.initial_state)
        test_loss, pred, final_big_frame_s, final_frame_s = sess.run(
            output_list, feed_dict=inp_dict)
        output = np.asarray(pred).reshape(-1)
        if i == 0:
            result = output
            continue
        result = crossfade(result, output, overlap)
    
    target = np.squeeze(wb_audio)[8: 8 + len(result)]
    pred = result
    
    l1 = l1_loss(pred, target)
    lsd = log_spectral_distance(pred, target, sess)
    write_wav(result, 16000, args.method + '.wav')
    
    print('Mean L1 loss = {}'.format(l1))
    print('Mean LSD = {}'.format(lsd))
    
    return

args = get_args()
evaluate(args)