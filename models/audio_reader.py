import fnmatch
import os
import random
import re
import threading
import librosa
import sys
import copy
import numpy as np
import tensorflow as tf

def randomize_files(files):
    files_idx = [i for i in xrange(len(files))]
    random.shuffles(files_idx)
    for idx in xrange(len(files)):
        yield files[files_idx[idx]]

def find_files(directory, pattern='*.wav'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_generic_audio(directory, sample_rate):
    files = find_files(directory)
    print('Files length: {}'.format(len(files)))
    randomized_files = randomize_files(files)
    for filename in files:
        wb_filename = filename.replace('nb', 'wb')
        # Yield both nb_audio and wb_audio given filename
        print("Found: {}, {}".format(filename, wb_filename))
        nb_audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        wb_audio, _ = librosa.load(wb_filename, sr=sample_rate, mono=True)
        nb_audio = nb_audio.reshape(-1, 1)
        wb_audio = wb_audio.reshape(-1, 1)
        yield nb_audio, wb_audio, filename

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue'''

    def __init__(self,
                 nb_audio_dir,
                 wb_audio_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.nb_audio_dir = nb_audio_dir
        self.wb_audio_dir = wb_audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.nb_sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.wb_sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32'],
                                         shapes=[(None, 1), (None, 1)])
        self.enqueue = self.queue.enqueue(
            [self.nb_sample_placeholder, self.wb_sample_placeholder])

        nb_files = find_files(nb_audio_dir)
        wb_files = find_files(wb_audio_dir)
        if not nb_files:
            raise ValueError("No audio files found in '{}'".format(nb_audio_dir))
        if not wb_files:
            raise ValueError("No audio files found in '{}'".format(wb_audio_dir))
        return

    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        nb_audio_list = []
        wb_audio_list = []

        # load_generic_audio takes NB directory and yields nb_audio, wb_audio,
        # and nb_filename
        iterator = load_generic_audio(self.nb_audio_dir, self.sample_rate)
        for nb_audio, wb_audio, _ in iterator:
            nb_audio_list.append(nb_audio)
            wb_audio_list.append(wb_audio)
        print('Compiled audio')
        while not stop:
            for nb_audio_copy, wb_audio_copy in zip(nb_audio_list, wb_audio_list):
                nb_audio = copy.deepcopy(nb_audio_copy)
                wb_audio = copy.deepcopy(wb_audio_copy)
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    nb_audio = trim_silence(nb_audio[:, 0], self.silence_threshold)
                    nb_audio = nb_audio.reshape(-1, 1)
                    wb_audio = trim_silence(wb_audio[:, 0], self.silence_threshold)
                    wb_audio = wb_audio.reshape(-1, 1)
                    #if nb_audio.size ==  0 or wb_audio.size == 0:
                        #print('An audio file was dropped.')

                pad_elements = \
                               self.sample_size - 1 \
                               - (nb_audio.shape[0] + self.sample_size - 1) \
                               % self.sample_size
                nb_audio = np.concatenate(
                    [nb_audio, np.full((pad_elements, 1), 0.0, dtype='float32')],
                    axis=0)
                wb_audio = np.concatenate(
                    [wb_audio, np.full((pad_elements, 1), 0.0, dtype='float32')],
                    axis=0)

                if self.sample_size:
                    # Keep taking chunks of size sample_size
                    while len(nb_audio) > self.sample_size:
                        nb_piece = nb_audio[:self.sample_size, :]
                        wb_piece = wb_audio[:self.sample_size, :]
                        sess.run(self.enqueue,
                                 feed_dict={self.nb_sample_placeholder: nb_piece,
                                            self.wb_sample_placeholder: wb_piece})
                        nb_audio = nb_audio[self.sample_size:, :]
                        wb_audio = wb_audio[self.sample_size:, :]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.nb_sample_placeholder: nb_audio,
                                        self.wb_sample_placeholder: wb_audio})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads
