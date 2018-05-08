from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import sys
import time
import numpy as np
import scipy
import tensorflow as tf
import librosa
import fnmatch
from models import HRNN_GAN, Discriminator, AudioReader
from models import find_files, get_test_batches, average_gradients, load, save, optimizer_factory, scalar_summary
from tensorflow import AggregationMethod as aggreg
from tensorflow.python.client import timeline

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--nb_data_dir', type=str, required=True)
    parser.add_argument('--wb_data_dir', type=str, required=True)
    parser.add_argument('--test_nb_data_dir', type=str, required=True)
    parser.add_argument('--test_wb_data_dir', type=str, required=True)
    parser.add_argument('--logdir_root', type=str, required=True)
    parser.add_argument('--ckpt_every', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--sample_size', type=int, default=48000)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--l2_reg_strength', type=float, default=0.0)
    parser.add_argument('--silence_threshold', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys())
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--big_frame_size', type=int, required=True)
    parser.add_argument('--frame_size', type=int, required=True)
    parser.add_argument('--q_levels', type=int, required=True)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--n_rnn', type=int, choices=xrange(1,6), required=True)
    parser.add_argument('--emb_size', type=int, required=True)
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], required=True)
    parser.add_argument('--max_checkpoints', type=int, default=5)
    parser.add_argument('--d_learning_rate', type=float, required=True)
    parser.add_argument('--bias_D_conv', type=bool, default=True)
    parser.add_argument('--pretrain_num_steps', type=int, required=True)
    parser.add_argument('--update_d_every', type=int, required=True)
    return parser.parse_args()

def create_model(args):
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

def create_discriminator(args, name):
    discr = Discriminator(bias_D_conv=args.bias_D_conv,
                          name=name)
    return discr

def train():
    args = get_args()
    if args.l2_reg_strength == 0:
        args.l2_reg_strength = None
    logdir = os.path.join(args.logdir_root, 'train')
    logdir_test = os.path.join(args.logdir_root, 'test')
    logdir_d = os.path.join(args.logdir_root, 'discriminator')
    coord = tf.train.Coordinator()

    # Number of steps to train HRNN only
    pretrain_num_steps = args.pretrain_num_steps
    # Number of steps to train HRNN before updating discriminator
    update_d_every = args.update_d_every
    
    # Get testing files
    test_files = find_files(args.test_nb_data_dir)
    
    # Create inputs
    with tf.name_scope('create_inputs'):
        reader = AudioReader(args.nb_data_dir,
                             args.wb_data_dir,
                             coord,
                             sample_rate=args.sample_rate,
                             sample_size=args.sample_size,
                             silence_threshold=args.silence_threshold)
        nb_audio_batch, wb_audio_batch = \
            reader.dequeue(args.batch_size)

    # Create model
    net = create_model(args)
    discr = create_discriminator(args, name='discr')
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0),
        trainable=False)
    
    # Optimizers
    optim = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum)
    d_optim = tf.train.AdamOptimizer(
        args.d_learning_rate)

    # Set up placeholders and variables on each GPU
    tower_net_grads = []
    tower_net_grads_no_adv = []
    tower_d_grads = []
    losses = []
    losses_no_adv = []
    losses_adv = []
    d_losses = []
    wb_input_batch_rnn = []
    nb_input_batch_rnn = []
    samplernn_preds = []
    train_big_frame_state = []
    train_frame_state = []
    final_big_frame_state = []
    final_frame_state = []
    goals = []
    predictions = []
    for i in xrange(args.num_gpus):
        with tf.device('/gpu:%d' % (i)):
            # Create input placeholders
            nb_input_batch_rnn.append(
                tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]),
                            trainable=False,
                            name='nb_input_batch_rnn',
                            dtype=tf.float32))
            wb_input_batch_rnn.append(
                tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]),
                            trainable=False,
                            name='wb_input_batch_rnn',
                            dtype=tf.float32))
            # Create initial states
            train_big_frame_state.append(
                net.big_cell.zero_state(net.batch_size, tf.float32))
            final_big_frame_state.append(
                net.big_cell.zero_state(net.batch_size, tf.float32))
            train_frame_state.append(
                net.cell.zero_state(net.batch_size, tf.float32))
            final_frame_state.append(
                net.cell.zero_state(net.batch_size, tf.float32))
            # Target/prediction placeholders
            goals.append(
                tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]),
                            trainable=False,
                            name='targets',
                            dtype=tf.float32))
            predictions.append(
                tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]),
                            trainable=False,
                            name='predictions',
                            dtype=tf.float32))
    
    # Network output variables
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(args.num_gpus):
            with tf.device('/gpu:%d' % (i)):
                with tf.name_scope('TOWER_%d' % i) as scope:
                    print("Creating model on GPU:%d" % i)
                    
                    # SampleRNN outputs
                    loss, final_big_frame_state[i], final_frame_state[i], \
                        goals[i], predictions[i] = \
                            net.loss_SampleRNN(
                                nb_input_batch_rnn[i],
                                wb_input_batch_rnn[i],
                                train_big_frame_state[i],
                                train_frame_state[i],
                                l2_reg_strength=args.l2_reg_strength)
                    
                    # Discriminator inputs
                    bfs = net.big_frame_size
                    predictions[i] = tf.reshape(
                        predictions[i],
                        [net.batch_size, net.seq_len-bfs, 1])
                    d_rl_input = tf.concat(
                        [tf.cast(wb_input_batch_rnn[i][:, :-bfs, :], dtype=tf.int32), 
                         tf.cast(nb_input_batch_rnn[i][:, :-bfs, :], dtype=tf.int32)],
                        2)
                    d_fk_input = tf.concat(
                        [tf.cast(predictions[i], dtype=tf.int32),
                         tf.cast(nb_input_batch_rnn[i][:, :-bfs, :], dtype=tf.int32)],
                        2)
                    d_rl_input = tf.cast(d_rl_input, dtype=tf.float32)
                    d_fk_input = tf.cast(d_fk_input, dtype=tf.float32)
                    
                    # Discriminator outputs
                    d_rl_logits = discr.logits_Discriminator(
                        d_rl_input, reuse=False)
                    d_fk_logits = discr.logits_Discriminator(
                        d_fk_input, reuse=True)
                    
                    # Discriminator loss
                    d_rl_loss = tf.reduce_mean(
                        tf.squared_difference(d_rl_logits, 1.0))
                    d_fk_loss = tf.reduce_mean(
                        tf.squared_difference(d_fk_logits, 0.0))
                    d_loss = d_rl_loss + d_fk_loss
                    d_losses.append(d_loss)
                    
                    # SampleRNN loss
                    net_adv_loss = tf.reduce_mean(
                        tf.squared_difference(d_fk_logits, 1.0))
                    net_loss = net_adv_loss + loss
                    losses.append(net_loss)
                    losses_no_adv.append(loss)
                    losses_adv.append(net_adv_loss)

                    # Scalar summaries
                    d_rl_loss_sum = scalar_summary('d_rl_loss', d_rl_loss)
                    d_fk_loss_sum = scalar_summary('d_fk_loss', d_fk_loss)
                    d_loss_sum = scalar_summary('d_loss', d_loss)
                    net_loss_sum = scalar_summary(
                        'samplernn_loss', net_loss)
                    net_loss_adv_sum = scalar_summary(
                        'samplernn_adv_loss', net_adv_loss)
                    
                    # Get trainable vars
                    net_trainable = tf.trainable_variables(
                        scope='SampleRNN')
                    d_trainable = tf.trainable_variables(
                        scope='Discriminator')
                    
                    # Gradients
                    gradients = optim.compute_gradients(
                        net_loss, net_trainable,
                        aggregation_method=aggreg.EXPERIMENTAL_ACCUMULATE_N)
                    gradients_no_adv = optim.compute_gradients(
                        loss, net_trainable,
                        aggregation_method=aggreg.EXPERIMENTAL_ACCUMULATE_N)
                    d_gradients = d_optim.compute_gradients(
                        d_loss, d_trainable,
                        aggregation_method=aggreg.EXPERIMENTAL_ACCUMULATE_N)
                    
                    tower_net_grads.append(gradients)
                    tower_net_grads_no_adv.append(gradients_no_adv)
                    tower_d_grads.append(d_gradients)
                    
                    tf.get_variable_scope().reuse_variables()
                    
    # Gradients              
    net_grad_vars = average_gradients(tower_net_grads)
    net_grad_vars_no_adv = average_gradients(tower_net_grads_no_adv)
    d_grad_vars = average_gradients(tower_d_grads)
    
    # Clip gradients
    grads, vars = zip(*net_grad_vars)
    grads_no_adv, vars = zip(*net_grad_vars_no_adv)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grads_clipped_no_adv, _ = tf.clip_by_global_norm(grads_no_adv, 5.0)
    net_grad_vars = zip(grads_clipped, vars)
    net_grad_vars_no_adv = zip(grads_clipped_no_adv, vars)

    # Apply gradient ops
    apply_gradient_op = optim.apply_gradients(
        net_grad_vars, global_step=global_step)
    apply_gradient_op_no_adv = optim.apply_gradients(
        net_grad_vars_no_adv, global_step=global_step)
    d_apply_gradient_op = d_optim.apply_gradients(
        d_grad_vars, global_step=global_step)
    
    # ---------------------------------------------------------------
    # Start/continue training
    # ---------------------------------------------------------------
    writer = tf.summary.FileWriter(logdir)
    test_writer = tf.summary.FileWriter(logdir_test)
    writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()

    # Configure session
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load checkpoint
    saver = tf.train.Saver(var_list=net_trainable,
                           max_to_keep=args.max_checkpoints)
    d_saver = tf.train.Saver(var_list=d_trainable,
                             max_to_keep=args.max_checkpoints)
    try:
        saved_global_step = load(saver, sess, logdir)
        load(d_saver, sess, logdir_d)
        if saved_global_step is None: saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint.")
        raise

    # Start queue runners
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    # Train
    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps + 1):
            final_big_s = []
            final_s = []
            for g in xrange(args.num_gpus):
                # Initialize cells
                final_big_s.append(sess.run(net.big_initial_state))
                final_s.append(sess.run(net.initial_state))
                start_time = time.time()

            nb_inputs_list = []
            wb_inputs_list = []
            for _ in xrange(args.num_gpus):
                # Get input batches
                nb_inputs, wb_inputs = sess.run(
                    [nb_audio_batch, wb_audio_batch])
                nb_inputs_list.append(nb_inputs)
                wb_inputs_list.append(wb_inputs)

            loss_sum = 0
            d_loss_sum = 0
            loss_adv_sum = 0
            
            idx_begin = 0
            audio_length = args.sample_size - args.big_frame_size
            bptt_length = args.seq_len - args.big_frame_size
            stateful_rnn_length = audio_length / bptt_length
            output_list = [summaries,
                           losses,
                           losses_adv,
                           d_losses,
                           apply_gradient_op,
                           final_big_frame_state,
                           final_frame_state]
            output_list_no_adv = [summaries,
                                  losses_no_adv,
                                  losses_adv,
                                  d_losses,
                                  apply_gradient_op_no_adv,
                                  final_big_frame_state,
                                  final_frame_state]
            discr_output_list = [d_apply_gradient_op]

            for i in range(0, stateful_rnn_length):
                inp_dict = {}
                for g in xrange(args.num_gpus):
                    # Add seq_len samples as input for truncated BPTT
                    inp_dict[nb_input_batch_rnn[g]] = \
                        nb_inputs_list[g][:, idx_begin:idx_begin+args.seq_len, :]
                    inp_dict[wb_input_batch_rnn[g]] = \
                        wb_inputs_list[g][:, idx_begin:idx_begin+args.seq_len, :]
                    inp_dict[train_big_frame_state[g]] = final_big_s[g]
                    inp_dict[train_frame_state[g]] = final_s[g]
                idx_begin += args.seq_len - args.big_frame_size

                # Forward pass
                if (step < pretrain_num_steps):
                    # Train with L1
                    summary, loss_gpus, loss_adv_gpus, d_loss_gpus, _, final_big_s, final_s = \
                        sess.run(output_list_no_adv, feed_dict=inp_dict)
                else:
                    # Train with L1 + adversarial loss
                    summary, loss_gpus, loss_adv_gpus, d_loss_gpus, _, final_big_s, final_s = \
                        sess.run(output_list, feed_dict=inp_dict)
                       
                writer.add_summary(summary, step)
                for g in xrange(args.num_gpus):
                    loss_gpu = loss_gpus[g] / stateful_rnn_length
                    d_loss_gpu = d_loss_gpus[g] / stateful_rnn_length
                    loss_adv_gpu = loss_adv_gpus[g] / stateful_rnn_length
                    
                    loss_sum += loss_gpu / args.num_gpus
                    d_loss_sum += d_loss_gpu / args.num_gpus
                    loss_adv_sum += loss_adv_gpu / args.num_gpus
            duration = time.time() - start_time
            
            print('****** STEP {:d} ({:.3f} sec/step) ******'.format(step, duration))
            if (step < pretrain_num_steps):
                print('[SampleRNN] L1 loss = {:.3f}'.format(loss_sum))
            else:
                print('[SampleRNN] L1 + adv loss = {:.3f}'.format(loss_sum))
            print('[SampleRNN] adv loss = {:.3f}'.format(loss_adv_sum))
            print('[Discriminator] L2 loss = {:.3f}'.format(d_loss_sum))
            
            if (step >= pretrain_num_steps) and (step % update_d_every == 0):
                # Update discriminator parameters
                print('Updating discriminator parameters...')
                _ = sess.run(discr_output_list, feed_dict=inp_dict)
            
            if step % args.ckpt_every == 0:
                # Save models
                save(saver, sess, logdir, step)
                save(d_saver, sess, logdir_d, step)
                last_saved_step = step
            
            if step % 20 == 0:
                # Test
                test_nb_inputs, test_wb_inputs = get_test_batches(
                    test_files, args.batch_size, args.sample_rate)
                test_output_list = [summaries,
                                    losses,
                                    final_big_frame_state,
                                    final_frame_state]
                test_output_list_no_adv = [summaries,
                                           losses_no_adv,
                                           final_big_frame_state,
                                           final_frame_state]
                
                loss_sum = 0
                idx_begin = 0
                audio_length = args.sample_size - args.big_frame_size
                bptt_length = args.seq_len - args.big_frame_size
                stateful_rnn_length = audio_length / bptt_length

                for i in range(0, stateful_rnn_length):
                    inp_dict = {}
                    for g in xrange(args.num_gpus):
                        # Add seq_len samples as input for truncated BPTT
                        inp_dict[nb_input_batch_rnn[g]] = \
                            nb_inputs_list[g][:, idx_begin:idx_begin+args.seq_len, :]
                        inp_dict[wb_input_batch_rnn[g]] = \
                            wb_inputs_list[g][:, idx_begin:idx_begin+args.seq_len, :]
                        inp_dict[train_big_frame_state[g]] = \
                            sess.run(net.big_initial_state)
                        inp_dict[train_frame_state[g]] = \
                            sess.run(net.initial_state)
                    idx_begin += args.seq_len - args.big_frame_size
                
                    # Forward pass
                    if (step < pretrain_num_steps):
                        summary, test_loss_gpus, final_big_s, final_s = \
                            sess.run(test_output_list_no_adv, feed_dict=inp_dict)
                    else:
                        summary, test_loss_gpus, final_big_s, final_s = \
                            sess.run(test_output_list, feed_dict=inp_dict)
                    test_writer.add_summary(summary, step)
                    
                    for g in xrange(args.num_gpus):
                        loss_gpu = test_loss_gpus[g] / stateful_rnn_length
                        loss_sum += loss_gpu / args.num_gpus
                print('Testing loss: {}'.format(loss_sum))

    except KeyboardInterrupt:
        print()
    finally:
        if step > last_saved_step:
            print('Saving HRNN model...')
            save(saver, sess, logdir, step)
            print('Saving discriminator model...')
            save(d_saver, sess, logdir_d, step)
        coord.request_stop()
        coord.join(threads)

train()