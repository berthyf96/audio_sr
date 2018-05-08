import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from .ops import mu_law_encode, mu_law_decode, log_mel_spectrograms

class HRNN(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.big_frame_size = args.big_frame_size
        self.frame_size = args.frame_size
        self.q_levels = args.q_levels
        self.rnn_type = args.rnn_type
        self.dim = args.dim
        self.n_rnn = args.n_rnn
        self.seq_len = args.seq_len
        self.emb_size = args.emb_size
        self.spec_loss_weight = args.spec_loss_weight
        self.l1_reg_strength = args.l1_reg_strength
        self.sample_rate = args.sample_rate
        
        # configure cells
        def single_cell():
            return tf.contrib.rnn.GRUCell(self.dim)
        if self.rnn_type == 'LSTM':
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.dim)
        self.cell = single_cell()
        self.big_cell = single_cell()
        if self.n_rnn > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)])
            self.big_cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)])
        self.initial_state = self.cell.zero_state(
            self.batch_size, tf.float32)
        self.big_initial_state = self.big_cell.zero_state(
            self.batch_size, tf.float32)
        
        # l1 regularizer
        if args.l1_reg_strength is not None:
            self.l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=args.l1_reg_strength)
        
        return
    
    def _create_network_BigFrame(self, num_steps, big_frame_state,
                                 big_input_sequences):
        bfs = self.big_frame_size
        fs = self.frame_size
        with tf.variable_scope('SampleRNN'):
            with tf.variable_scope('big_frame'):
                big_input_frames_shape = \
                    [tf.shape(big_input_sequences)[0],
                     tf.shape(big_input_sequences)[1] / bfs,
                     bfs]
                big_input_frames = tf.reshape(
                    big_input_sequences, big_input_frames_shape)
                big_input_frames = \
                    (big_input_frames / self.q_levels/2.0) - 1.0
                big_input_frames *= 2.0
                big_frame_outputs = []
                
                # weights
                big_frame_proj_weights = tf.get_variable(
                    'big_frame_proj_weights',
                    [self.dim, self.dim * bfs/fs],
                    dtype=tf.float32)
                
                with tf.variable_scope('big_frame_rnn'):
                    for time_step in range(num_steps):
                        if time_step > 0:
                            tf.get_variable_scope().reuse_variables()
                        # get output and state at this time step
                        (big_frame_output, big_frame_state) = self.big_cell(
                            big_input_frames[:, time_step, :],
                            big_frame_state)
                        big_frame_outputs.append(
                            math_ops.matmul(
                                big_frame_output,
                                big_frame_proj_weights))
                    final_big_frame_state = big_frame_state
                big_frame_outputs = tf.stack(big_frame_outputs)
                big_frame_outputs = tf.transpose(
                    big_frame_outputs, perm=[1,0,2])
                big_frame_outputs_shape = \
                    [tf.shape(big_frame_outputs)[0],
                     tf.shape(big_frame_outputs)[1] * bfs/fs,
                     -1]
                big_frame_outputs = tf.reshape(
                    big_frame_outputs, big_frame_outputs_shape)
                return big_frame_outputs, final_big_frame_state
    
    def _create_network_Frame(self, num_steps, big_frame_outputs,
                              frame_state, input_sequences):
        fs = self.frame_size
        with tf.variable_scope('SampleRNN'):
            with tf.variable_scope('frame'):
                input_frames_shape = \
                    [tf.shape(input_sequences)[0],
                     tf.shape(input_sequences)[1] / fs,
                     fs]
                input_frames = tf.reshape(
                    input_sequences, input_frames_shape)
                input_frames = (input_frames / self.q_levels/2.0) - 1.0
                input_frames *= 2.0
                frame_outputs = []
                
                # weights
                frame_proj_weights = tf.get_variable(
                    'frame_proj_weights',
                    [self.dim, self.dim * fs],
                    dtype=tf.float32)
                frame_cell_proj_weights = tf.get_variable(
                    'frame_cell_proj_weights',
                    [fs, self.dim],
                    dtype=tf.float32)
                
                with tf.variable_scope('frame_rnn'):
                    for time_step in range(num_steps):
                        if time_step > 0:
                            tf.get_variable_scope().reuse_variables()
                        # get input
                        cell_input = tf.reshape(
                            input_frames[:, time_step, :],
                            [-1, self.frame_size])
                        cell_input = math_ops.matmul(
                            cell_input, frame_cell_proj_weights)
                        # add big frame output to input
                        bf_output = tf.reshape(
                            big_frame_outputs[:, time_step, :],
                            [-1, self.dim])
                        cell_input = tf.add(cell_input, bf_output)
                        # get outputs
                        (frame_cell_output, frame_state) = self.cell(
                            cell_input, frame_state)
                        frame_outputs.append(
                            math_ops.matmul(
                                frame_cell_output, frame_proj_weights))
                    final_frame_state = frame_state
                frame_outputs = tf.stack(frame_outputs)
                frame_outputs = tf.transpose(frame_outputs, perm=[1,0,2])
                frame_outputs_shape = \
                    [tf.shape(frame_outputs)[0],
                     tf.shape(frame_outputs)[1] * fs,
                     -1]
                frame_outputs = tf.reshape(frame_outputs, frame_outputs_shape)
                return frame_outputs, final_frame_state
            
    def _create_network_Sample(self, frame_outputs, sample_input_sequences):
        with tf.variable_scope('SampleRNN'):
            with tf.variable_scope('sample'):
                sample_shape = \
                    [tf.shape(sample_input_sequences)[0],
                     tf.shape(sample_input_sequences)[1] * self.emb_size,
                     1]
                    
                # embedding layer
                embedding = tf.get_variable(
                    'embedding', [self.q_levels, self.emb_size])
                sample_input_sequences = embedding_ops.embedding_lookup(
                    embedding, tf.reshape(sample_input_sequences, [-1]))
                sample_input_sequences = tf.reshape(
                    sample_input_sequences, sample_shape)
                
                # convolution
                filter_initializer = tf.contrib.layers.xavier_initializer_conv2d()
                sample_filter_shape = [self.emb_size*2, 1, self.dim]
                sample_filter = tf.get_variable(
                    'sample_filter', sample_filter_shape,
                    initializer=filter_initializer)
                out = tf.nn.conv1d(sample_input_sequences,
                                   sample_filter,
                                   stride=self.emb_size,
                                   padding='VALID',
                                   name='sample_conv')
                out = tf.add(out, frame_outputs)
                
                # multilayer perceptron
                sample_mlp1_weights = tf.get_variable(
                    'sample_mlp1', [self.dim, self.dim], dtype=tf.float32)
                sample_mlp2_weights = tf.get_variable(
                    'sample_mlp2', [self.dim, self.dim], dtype=tf.float32)
                sample_mlp3_weights = tf.get_variable(
                    'sample_mlp3', [self.dim, 1], dtype=tf.float32)
                out = tf.reshape(out, [-1, self.dim])
                out = math_ops.matmul(out, sample_mlp1_weights)
                out = tf.nn.relu(out)
                out = math_ops.matmul(out, sample_mlp2_weights)
                out = tf.nn.relu(out)
                out = math_ops.matmul(out, sample_mlp3_weights)
                out = tf.reshape(
                    out, [-1, sample_shape[1]/self.emb_size - 1, 1])
                out = tf.multiply(tf.sigmoid(out), (self.q_levels - 1))
                return out
    
    def _create_network_SampleRNN(self, train_big_frame_state, 
                                  train_frame_state):
        bfs = self.big_frame_size
        fs = self.frame_size
        with tf.name_scope('SampleRNN'):
            # big frame network
            big_input_sequences = self.encoded_nb_input_rnn[:, :-bfs, :]
            big_input_sequences = tf.cast(big_input_sequences, tf.float32)
            big_frame_num_steps = (self.seq_len - bfs) / bfs
            big_frame_outputs, final_big_frame_state = \
                self._create_network_BigFrame(
                    num_steps=big_frame_num_steps,
                    big_frame_state=train_big_frame_state,
                    big_input_sequences=big_input_sequences)
            
            # frame network
            input_sequences = self.encoded_nb_input_rnn[:, bfs-fs:-fs, :]
            input_sequences = tf.cast(input_sequences, tf.float32)
            frame_num_steps = (self.seq_len - bfs) / fs
            frame_outputs, final_frame_state = \
                self._create_network_Frame(
                    num_steps=frame_num_steps,
                    big_frame_outputs=big_frame_outputs,
                    frame_state=train_frame_state,
                    input_sequences=input_sequences)
                
            # sample
            sample_input_sequences = self.encoded_nb_input_rnn[:, bfs-fs:-1, :]
            sample_output = self._create_network_Sample(
                frame_outputs, sample_input_sequences=sample_input_sequences)
            
            return sample_output, final_big_frame_state, final_frame_state
        
    def forward(self, nb_input_batch_rnn, wb_input_batch_rnn,
                train_big_frame_state, train_frame_state,
                l1_reg_strength=None, inference_only=False):
        bfs = self.big_frame_size
        with tf.name_scope('forward'):
            self.encoded_nb_input_rnn = mu_law_encode(
                nb_input_batch_rnn, self.q_levels)
            self.encoded_wb_input_rnn = mu_law_encode(
                wb_input_batch_rnn, self.q_levels)
            raw_output, final_big_frame_state, final_frame_state = \
                self._create_network_SampleRNN(
                    train_big_frame_state, train_frame_state)
            with tf.name_scope('total_loss'):
                # ---------------------------
                # L1 loss
                # ---------------------------
                target_output_rnn = \
                    self.encoded_wb_input_rnn[:, bfs:, :]
                target_output_rnn = tf.reshape(
                    target_output_rnn, [self.batch_size, -1, 1])
                target_output_rnn = tf.cast(target_output_rnn, dtype=tf.float32)
                prediction = raw_output
                reg_penalty = 0
                if l1_reg_strength is not None:
                    print('Applying L1 regularization')
                    reg_penalty = tf.contrib.layers.apply_regularization(
                        self.l1_regularizer, tf.trainable_variables())
                l1_loss = tf.losses.absolute_difference(
                    target_output_rnn, prediction)
                l1_loss = tf.reduce_mean(l1_loss) + reg_penalty
                tf.summary.scalar('l1_loss', l1_loss)
                
                # ---------------------------
                # spectral loss
                # ---------------------------
                # mu-law decode prediction
                pred_signals = tf.squeeze(raw_output, axis=2)
                pred_signals = mu_law_decode(pred_signals, self.q_levels)
                gt_signals = tf.squeeze(wb_input_batch_rnn, axis=2)
                
                # get log-mel spectrograms of prediction and ground truth
                prediction_spec = log_mel_spectrograms(
                    pred_signals, self.sample_rate)
                gt_spec = log_mel_spectrograms(
                    gt_signals[:, bfs:], self.sample_rate)
                
                # compute L2 loss between log-mel spectrograms
                spec_loss = tf.squared_difference(gt_spec, prediction_spec)
                spec_loss = tf.reduce_mean(spec_loss)
                tf.summary.scalar('spectral_loss', spec_loss)
                
                # ---------------------------
                # total loss
                # ---------------------------
                total_loss = l1_loss + (self.spec_loss_weight * spec_loss)
                tf.summary.scalar('total_loss', total_loss)
                
                if inference_only:
                    return total_loss, pred_signals, final_big_frame_state, final_frame_state
                else:
                    return total_loss, final_big_frame_state, final_frame_state