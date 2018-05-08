from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from .ops import downconv, leakyrelu, prelu, conv1d
import numpy as np

class Discriminator(object):
    
    def __init__(self, bias_D_conv, name):
        self.bias_D_conv = bias_D_conv
        self.d_num_fmaps = \
            [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.name = name
        
    def logits_Discriminator(self, d_input, reuse=False):
        in_dims = d_input.get_shape().as_list()
        hi = d_input
        if len(in_dims) == 2:
            hi = tf.expand_dims(d_input, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Discriminator input must be 2-D or 3-D')
        
        with tf.variable_scope('Discriminator'):
            with tf.variable_scope(self.name) as scope:

                if reuse: 
                    scope.reuse_variables()
                def disc_block(block_idx, input_, kwidth, 
                               nfmaps, bnorm, activation,
                               pooling=2):
                    with tf.variable_scope('d_block_{}'.format(block_idx)):
                        bias_init = None
                        if self.bias_D_conv:
                            bias_init = tf.constant_initializer(0.)
                        downconv_init = \
                            tf.truncated_normal_initializer(stddev=0.02)

                        # downconvolution
                        hi_a = downconv(input_, nfmaps, kwidth=kwidth,
                                        pool=pooling, init=downconv_init,
                                        bias_init=bias_init)

                        # VBN

                        # activation
                        if activation == 'leakyrelu':
                            hi = leakyrelu(hi_a)
                        elif activation == 'relu':
                            hi = tf.nn.relu(hi_a)
                        else:
                            raise ValueError('Unrecognized activation {}'
                                             'in D'.format(activation))
                        return hi

                # [removed] apply input noisy layer to real and fake samples

                for block_idx, fmaps in enumerate(self.d_num_fmaps):
                    hi = disc_block(block_idx, hi, 31,
                                    self.d_num_fmaps[block_idx],
                                    True, 'leakyrelu')
                if not reuse:
                    print('Discriminator deconved shape: ', hi.get_shape())
                #hi_f = flatten(hi)
                d_logit_out = conv1d(
                    hi, kwidth=1, num_kernels=1,
                    init=tf.truncated_normal_initializer(stddev=0.02),
                    name='logits_conv')
                d_logit_out = tf.squeeze(d_logit_out)
                d_logit_out = tf.expand_dims(d_logit_out, 1)
                d_logit_out = fully_connected(d_logit_out, 1, activation_fn=None)

                if not reuse:
                    print('Discriminator output shape: ', d_logit_out.get_shape())
                    print('*****************************')
                return d_logit_out