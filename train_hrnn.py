from __future__ import print_function
import tensorflow as tf
from tensorflow import AggregationMethod as aggreg
from tensorflow.python.client import timeline
from models import HRNN, AudioReader
from models import optimizer_factory, find_files, get_test_batches, average_gradients, load, save
import argparse
import os
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--nb_data_dir', type=str, required=True)
    parser.add_argument('--wb_data_dir', type=str, required=True)
    parser.add_argument('--test_nb_data_dir', type=str, required=True)
    parser.add_argument('--test_wb_data_dir', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--ckpt_every', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--sample_size', type=int, default=48000)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--l1_reg_strength', type=float, default=0.0)
    parser.add_argument('--silence_threshold', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys())
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seq_len', type=int, default=520)
    parser.add_argument('--big_frame_size', type=int, default=8)
    parser.add_argument('--frame_size', type=int, default=2)
    parser.add_argument('--q_levels', type=int, default=256)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--n_rnn', type=int, choices=xrange(1,6), default=1)
    parser.add_argument('--emb_size', type=int, default=256)
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], default='LSTM')
    parser.add_argument('--max_checkpoints', type=int, default=10)
    parser.add_argument('--spec_loss_weight', type=float, required=True)
    return parser.parse_args()

def create_model(args):
    '''Set up model, global step, and optimizer'''
    model = HRNN(args)
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0),
        trainable=False)
    optim = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate, momentum=args.momentum)
    return model, global_step, optim
    
def train():
    # -----------------------------------------
    # SETUP
    # -----------------------------------------
    args = get_args()
    seq_len = args.seq_len
    if args.l1_reg_strength == 0.0:
        args.l1_reg_strength = None 
    logdir = os.path.join(args.logdir, 'train')
    logdir_test = os.path.join(args.logdir, 'test')
    coord = tf.train.Coordinator()
    # get test files
    test_files = find_files(args.test_nb_data_dir)
    # create inputs
    with tf.name_scope('create_inputs'):
        reader = AudioReader(args.nb_data_dir,
                             args.wb_data_dir,
                             coord,
                             sample_rate=args.sample_rate,
                             sample_size=args.sample_size,
                             silence_threshold=args.silence_threshold)
        nb_audio_batch, wb_audio_batch = reader.dequeue(args.batch_size)
    # create model
    net, global_step, optim = create_model(args)
    
    # set up placeholders and variables on each GPU
    nb_input_batch = []
    wb_input_batch = []
    train_big_frame_state = []
    train_frame_state = []
    final_big_frame_state = []
    final_frame_state = []
    losses = []
    tower_grads = []
    for i in range(args.num_gpus):
        with tf.device('/gpu:%d' % i):
            # create input placeholders
            nb_input_batch.append(tf.Variable(
                tf.zeros([net.batch_size, seq_len, 1]),
                trainable=False,
                name='nb_input_batch_rnn',
                dtype=tf.float32))
            wb_input_batch.append(tf.Variable(
                tf.zeros([net.batch_size, seq_len, 1]),
                trainable=False,
                name='wb_input_batch_rnn',
                dtype=tf.float32))
            # create initial states
            train_big_frame_state.append(
                net.big_cell.zero_state(net.batch_size, tf.float32))
            final_big_frame_state.append(
                net.big_cell.zero_state(net.batch_size, tf.float32))
            train_frame_state.append(
                net.cell.zero_state(net.batch_size, tf.float32))
            final_frame_state.append(
                net.cell.zero_state(net.batch_size, tf.float32))
    
    # network output variables
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('TOWER_%d' % i) as scope:
                    # create variables
                    print('Creating model on GPU:%d' % i)
                    loss, final_big_frame_state[i], final_frame_state[i] = \
                        net.forward(nb_input_batch[i],
                                    wb_input_batch[i],
                                    train_big_frame_state[i],
                                    train_frame_state[i],
                                    l1_reg_strength=args.l1_reg_strength)
                    tf.get_variable_scope().reuse_variables()
                    losses.append(loss)
                    # reuse variables for the next tower
                    trainable = tf.trainable_variables()
                    gradients = optim.compute_gradients(
                        loss, trainable,
                        aggregation_method=aggreg.EXPERIMENTAL_ACCUMULATE_N)
                    tower_grads.append(gradients)
    
    # backpropagation
    grad_vars = average_gradients(tower_grads)
    grads, vars = zip(*grad_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grad_vars = zip(grads_clipped, vars)
    apply_gradient_op = optim.apply_gradients(
        grad_vars, global_step=global_step)
    
    # configure session
    writer = tf.summary.FileWriter(logdir)
    test_writer = tf.summary.FileWriter(logdir_test)
    writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # load checkpoint
    saver = tf.train.Saver(
        var_list=tf.trainable_variables(),
        max_to_keep=args.max_checkpoints)
    try:
        saved_global_step = load(saver, sess, logdir)
        if saved_global_step is None: saved_global_step = -1
    except:
        raise ValueError('Something went wrong while restoring checkpoint')
        
    # start queue runners
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    
    # -----------------------------------------
    # TRAIN + VAL
    # -----------------------------------------
    print('Starting training...')
    step = None
    last_saved_step = saved_global_step
    for step in range(saved_global_step + 1, args.num_steps + 1):
        # initialize cells
        final_big_s = []
        final_s = []
        for g in range(args.num_gpus):
            final_big_s.append(sess.run(net.big_initial_state))
            final_s.append(sess.run(net.initial_state))
            start_time = time.time()
            
        # get input batches
        nb_inputs_list = []
        wb_inputs_list = []
        for _ in range(args.num_gpus):
            nb_inputs, wb_inputs = sess.run(
                [nb_audio_batch, wb_audio_batch])
            nb_inputs_list.append(nb_inputs)
            wb_inputs_list.append(wb_inputs)
        
        # run BPTT
        audio_length = args.sample_size - args.big_frame_size
        bptt_length = seq_len - args.big_frame_size
        stateful_rnn_length = audio_length / bptt_length
        loss_sum = 0
        idx_begin = 0
        output_list = [summaries,
                       losses,
                       apply_gradient_op,
                       final_big_frame_state,
                       final_frame_state]
        for i in range(stateful_rnn_length):
            inp_dict = {}
            for g in range(args.num_gpus):
                # add seq_len samples as input for truncated BPTT
                inp_dict[nb_input_batch[g]] = \
                    nb_inputs_list[g][:, idx_begin:idx_begin+seq_len, :]
                inp_dict[wb_input_batch[g]] = \
                    wb_inputs_list[g][:, idx_begin:idx_begin+seq_len, :]
                inp_dict[train_big_frame_state[g]] = final_big_s[g]
                inp_dict[train_frame_state[g]] = final_s[g]
            idx_begin += seq_len - args.big_frame_size
            
            # forward pass
            summary, loss_gpus, _, final_big_s, final_s = \
                sess.run(output_list, feed_dict=inp_dict)
            writer.add_summary(summary, step)
            for g in range(args.num_gpus):
                loss_gpu = loss_gpus[g] / stateful_rnn_length
                loss_sum += loss_gpu / args.num_gpus
        duration = time.time() - start_time
        print('Step {:d}: loss = {:.3f}, ({:.3f} sec/step)'.format(
            step, loss_sum, duration))
        
        if step % args.ckpt_every == 0:
            save(saver, sess, logdir, step)
            last_saved_step = step
        
        # validation
        if step % 20 == 0:
            print('Testing...')
            test_nb_inputs, test_wb_inputs = get_test_batches(
                test_files, args.batch_size, args.sample_rate)
            test_output_list = [summaries,
                                losses,
                                final_big_frame_state,
                                final_frame_state]
            loss_sum = 0
            idx_begin = 0
            audio_length = args.sample_size - args.big_frame_size
            bptt_length = seq_len - args.big_frame_size
            stateful_rnn_length = audio_length / bptt_length
            for i in range(stateful_rnn_length):
                inp_dict = {}
                for g in range(args.num_gpus):
                    inp_dict[nb_input_batch[g]] = \
                        nb_inputs_list[g][:, idx_begin:idx_begin+seq_len, :]
                    inp_dict[wb_input_batch[g]] = \
                        wb_inputs_list[g][:, idx_begin:idx_begin+seq_len, :]
                    inp_dict[train_big_frame_state[g]] = \
                        sess.run(net.big_initial_state)
                    inp_dict[train_frame_state[g]] = \
                        sess.run(net.initial_state)
                idx_begin += seq_len - args.big_frame_size
                # forward pass
                summary, test_loss, final_big_s, final_s = \
                    sess.run(test_output_list, feed_dict=inp_dict)
                test_writer.add_summary(summary, step)
                for g in range(args.num_gpus):
                    loss_gpu = loss_gpus[g] / stateful_rnn_length
                    loss_sum += loss_gpu / args.num_gpus
            print('Step {:d}: val loss = {:.3f}'.format(step, loss_sum))
    
    # done training
    coord.request_stop()
    coord.join(threads)
    return

train()