import os
import re
import sys
import threading
import time
from collections import defaultdict, namedtuple
from glob import glob
from typing import Dict, List, Union, Tuple, Iterable

import numpy as np
import tensorflow as tf

# Global variables
logger = None  # Defined after definition of Utils class


class Markov:
    """Class for representing a k-character substring and all of the possible chars that come after it"""

    def __init__(self, substring: str) -> None:
        """Creates a Markov object for a given substring"""
        self.substring = substring
        self._count = 1
        self._suffixes = defaultdict(int)

    def count(self) -> int:
        """Returns the count of this Markov object's substring"""
        return self._count

    def substring(self) -> str:
        """Returns this Markov object's substring"""
        return self.substring

    def add(self) -> None:
        """Adds an occurrence of this Markov object's substring"""
        self._count += 1

    def add_suffix(self, c: chr) -> None:
        """Adds an occurrence of a suffix to this Markov object"""
        self._suffixes[c] += 1

    def __str__(self) -> str:
        """Creates a string representation of this Markov object"""
        return "<Markov> '{}': {}".format(self.substring, dict(self._suffixes))

    def random(self) -> chr:
        """Chooses a random suffix based on probability distribution"""
        suffs = list(self._suffixes.keys())
        total = sum(self._suffixes[c] for c in suffs)
        return np.random.choice(suffs, p=[self._suffixes[c] / total for c in suffs])

    def suffixes(self) -> Dict[chr, int]:
        """Returns the suffix dict for this Markov's substring"""
        return dict(self._suffixes)


class MarkovChain:
    """Represents a Markov chain of given text input"""

    def __init__(self, regex: str=None, k=12) -> None:
        """Initializes a Markov chain to a given set of directories filled with text files

        Keyword Arguments:
            regex: The glob pattern to grab directories of text files to use (default cwd)
            k: The order of this Markov chain (default 12)
        """
        self._subs: Dict[str, Markov] = {}
        self._texts = set()
        self.k = k
        self.starts = []
        self.__add__(regex)

    def __add__(self, regex: str) -> None:
        """Adds text files to this Markov chain. Accepts globs to directory locations. Use '' for cwd"""
        for f in [_f for _f in Utils.get_files(regex) if _f not in self._texts]:
            self._texts.add(f)
            with open(f, 'r', encoding='utf-8') as io:
                text = io.read()
                for i in range(len(text) - (self.k + 1)):
                    # Grab a k-length substring starting at index i in the text
                    sub = text[i:i+self.k]
                    if i == 0:
                        self.starts.append(sub)
                    if sub not in self._subs:
                        self._subs[sub] = Markov(sub)
                    else:
                        self._subs[sub].add()
                    # Add the character directly after the substring as a suffix, or increase its counter by 1
                    self._subs[sub].add_suffix(text[i + self.k])

    def get_subs(self) -> List[str]:
        """Returns a list of all substrings in this Markov chain"""
        return list(self._subs.keys())

    def get_suffixes(self, substring: str, with_freq=False) -> Union[List[chr], Dict[chr, float]]:
        """Returns either a list of all possible suffixes of a substring in this Markov chain, or a dict containing
        those suffixes mapped to their frequencies"""
        if substring not in self._subs:
            MarkovChain._missing(substring)
        suffs = self._subs[substring].suffixes()
        if with_freq:
            total = sum(suffs[c] for c in suffs)
            return {c: suffs[c] / total for c in suffs}
        else:
            return list(suffs.keys())

    def random_suffix(self, substring: str) -> chr:
        if substring not in self._subs:
            MarkovChain._missing(substring)
        return self._subs[substring].random()

    def get_start(self) -> str:
        """Returns a random starting string (the first k characters in any of the files used)"""
        return np.random.choice(self.starts)

    def __len__(self) -> int:
        return sum(m.count() for m in self._subs.values())

    @staticmethod
    def _missing(substring: str) -> None:
        """Raises error for missing substring in this Markov chain"""
        raise KeyError('{} not in this Markov chain'.format(substring))


class Utils:
    """Class for Markov chain utilities"""
    _num_check = re.compile('[0-9]+((\.[0-9]+)?)')
    INSTANCE_DELIMITER = '\n\n$||||$\n\n'

    @staticmethod
    def get_files(regex: str=None) -> List[str]:
        """Globs together directories and pulls all the files out of them"""
        files = []
        for path in glob(regex if regex is not None else r'*'):
            # Skip all non-directories
            if not os.path.isdir(path):
                continue
            # List all the files in the directory being used
            for f in os.listdir(path):
                # Skip nested directories (only use plain files)
                if os.path.isdir(f):
                    continue
                files.append(path + '/' + f)
        return files

    @staticmethod
    def is_num(v) -> bool:
        """Tells if a variable is a number"""
        r = Utils._num_check.match(str(v))
        return r is not None and None not in r.groups()

    @staticmethod
    def safe_path(path: str, is_dir=False) -> str:
        """Creates a version of the path such that no files/directories will be overwritten"""
        possible = re.compile(r'^' + path + r'(\([0-9]+\))?$')
        # Get a list of all files/dirs in the cwd depending on whether or not is_dir is true
        files = [f for f in glob('*') if possible.match(f) is not None and (is_dir == os.path.isdir(f))]
        last = max(int(f[-2]) if f.endswith(')') else 0 for f in files) if len(files) > 0 else -1
        return path + ('' if last == -1 else '({})'.format(str(last + 1)))

    class Logger:
        """Class used for centralized logging so files are written cleanly and efficiently"""
        _log_types = ['ARCHITECTURE', 'LOSS', 'EVAL', 'MISC', 'GLOBAL']
        categories = namedtuple('LogTypes', _log_types)._make(_log_types)

        def __init__(self, interval=60) -> None:
            """Constructor

            Keyword Arguments:
                interval: The amount of seconds the logger will wait before the message queue is flushed to the disk
            """
            self.message_queue = defaultdict(list)
            self._waiting = set()
            self._write = False
            self.summaries = {}
            self.refresh_period = interval
            self.thread = threading.Thread(target=self.flush, args=())
            self.thread.daemon = True
            self.thread.name = 'Logger'
            self.thread.start()

        def log(self, message: str, file: str) -> None:
            """Stores a log message in memory to be written to disk later"""
            # Wait for any current writes to finish
            while self._write:
                continue
            self.message_queue[file].append(message)
            self._waiting.add(file)

        def flush(self) -> None:
            """Flushes the message queue"""
            self._write = True
            self._write_to_disk()
            self._write = False
            try:
                time.sleep(self.refresh_period)
            except KeyboardInterrupt:
                print('Flushing log...')
                self._write_to_disk()
                sys.exit(1)

        def _write_to_disk(self):
            """Writes the contents of the queue to disk"""
            for file in self._waiting:
                with open(file, 'a+', encoding='utf-8') as log:
                    log.write('\n'.join(self.message_queue[file]))
                self.message_queue[file].clear()
            self._waiting.clear()

        def add_summary(self, category: str, tracking: tf.Tensor) -> None:
            """Adds a summary op to the logger"""
            self.summaries[category] = tracking

    @staticmethod
    def unicode_range(encoding: str) -> int:
        """Returns the range of values for a unicode encoding"""
        r = re.match(r'utf-[0-9]+', encoding)
        if not (r is not None and None not in r.groups()):
            return 0
        return 1 << int(re.search(r'[0-9]+', encoding).group(0))


def set_logger(new: Utils.Logger) -> Utils.Logger:
    global logger
    logger = new
    return logger


set_logger(Utils.Logger())


class TextNet:
    """Class that represents a baseline neural network used for text generation/classification"""
    _unnamed = 0
    names = set()
    he_init = tf.variance_scaling_initializer()
    EXTERNAL = 'External'

    def __init__(self,
                 learning_rate=0.001,
                 logging: bool = False,
                 verbose=False, gm_time=False,
                 log_file: str = None,
                 exclude: Iterable[str]=(),
                 encoding='utf-8',
                 name=None) -> None:
        """Creates an instance of an Text Neural Network

        Keyword Arguments:
            learning_rate: The learning rate of this RNN (default 0.001)
            logging: Tells if this TextNet is to be logged to a file
            verbose: Tells if updates to this neural network will be printed to the console
            gm_time: Sets logging to Greenwich meantime rather than local time
            log_file: The name of the log file for this TextNet
            exclude: Categories to not be logged. For options, see Utils.Logger._log_types
            encoding: The type of encoding for the characters used by this TextRNN
            name: The name of this neural network
        """
        self._verbose = verbose
        self._learning_rate = float(learning_rate)
        self._name = str(name) if name is not None else None
        if name is None:
            self._name = 'unnamed_' + self.__class__.__name__ + '-' + str(self.__class__._unnamed)
            self.__class__._unnamed += 1
        if self._name in self.__class__.names:
            raise ValueError('Cannot have different variables of same type with same name: {}'.format(self._name))
        self.__class__.names.add(self._name)
        self._log_file = Utils.safe_path('logs/' + log_file if log_file is not None else self._name + '-log')
        self._logging = {_type: (_type not in exclude) for _type in Utils.Logger.categories}
        self._logging[Utils.Logger.categories.GLOBAL] = logging
        self._log_time = time.gmtime if gm_time else time.localtime
        self.encoding = encoding
        self.x = None
        self.y = None
        self.last_added = None
        self.loss = None
        self.accuracy = None
        with tf.name_scope(TextNet.EXTERNAL):
            self.training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self._closed = False
        global logger
        self.logger = logger
        self.log('Initialize ' + str(self))

    def get_name(self) -> str:
        """Returns the generic name of this RNN"""
        return self._name

    def log(self, message: str = None, category: str=None) -> None:
        """Logs a message or just the time if no message is provided. Covers both printing and file logging"""
        out = time.strftime('[%a, %d %b %Y %H:%M:%S] ', self._log_time()) + '[' + self.get_name() + '] ' \
            + (message if message is not None else '')
        if self._verbose and self._logging[category]:
            print(out)
        if self._logging[Utils.Logger.categories.GLOBAL] and self._logging[category]:
            self.logger.log(out, self._log_file)

    def __str__(self) -> str:
        """Returns a string representation of this TextRNN object (includes pertinent information)"""
        return self.__class__.__name__ + '(name={}, learning_rate={}'.format(
            self._name, self._learning_rate)

    def add_input_layer(self, input_type=tf.float32, input_shape=None,
                        output_type=tf.float32, output_shape=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Adds an input layer to this network's architecture"""
        self._check_closed()
        with tf.name_scope(self.get_name()):
            with tf.name_scope('Input'):
                self.x = tf.placeholder(input_type, shape=input_shape, name='X')
                self.y = tf.placeholder(output_type, shape=output_shape, name='y')
        self.last_added = self.x
        self.log('Initialize input layer', Utils.Logger.categories.ARCHITECTURE)
        return self.x, self.y

    def add_dense_layer(self, n_neurons, kernel_initializer=None, scope_name='', *args, **kwargs) -> tf.Tensor:
        """Adds a dense layer to this neural network

        Keyword Arguments:
            n_neurons: The number of neurons in this layer
            kernel_initializer: The random initializer for the layer (default He initialization)
            scope_name: The name of the greater layer (default 'Hidden')
        """
        self._check_closed()
        if kernel_initializer is None:
            kernel_initializer = TextNet.he_init
        with tf.name_scope(self.get_name()):
            with tf.name_scope(scope_name):
                self.last_added = tf.layers.dense(self.last_added, n_neurons, kernel_initializer=kernel_initializer,
                                                  name='Fully_Connected', *args, **kwargs)
        self.log('Add dense layer under name ' + scope_name, Utils.Logger.categories.ARCHITECTURE)
        return self.last_added

    def add_batch_norm_layer(self, scope_name='', *args, **kwargs) -> tf.Tensor:
        """Add a batch normalization layer to the model"""
        self._check_closed()
        with tf.name_scope(self.get_name()):
            with tf.name_scope(scope_name):
                self.last_added = tf.layers.batch_normalization(self.last_added, training=self.training,
                                                                name='Batch_Norm', *args, **kwargs)
        self.log('Add batch normalization layer under name ' + scope_name, Utils.Logger.categories.ARCHITECTURE)
        return self.last_added

    def activate(self, activate=tf.nn.selu, scope_name='', *args, **kwargs) -> tf.Tensor:
        """Adds an activation function to the most recently added layer"""
        self._check_closed()
        with tf.name_scope(self.get_name()):
            with tf.name_scope(scope_name):
                self.last_added = activate(self.last_added, *args, **kwargs)
        self.log('Activate {} with {}'.format(scope_name, activate.__name__), Utils.Logger.categories.ARCHITECTURE)
        return self.last_added

    def close(self, loss_fn=tf.reduce_mean, error_fn=tf.nn.sparse_softmax_cross_entropy_with_logits, *args, **kwargs) \
            -> tf.Tensor:
        """Closes the network and returns the loss op Tensor. args and kwargs will be passed directly to error_fn"""
        self._closed = True
        with tf.name_scope(self.get_name()):
            self.loss = loss_fn(error_fn(*args, **kwargs), name='loss')
            if self._logging[Utils.Logger.categories.LOSS]:
                self.logger.add_summary(Utils.Logger.categories.LOSS, tf.summary.scalar('loss', self.loss))
        return self.loss

    def add_eval_accuracy(self, logits=None, top_k=1) -> tf.Tensor:
        """Adds an accuracy evaluation op to the end of the network. Default logits are the last-added layer"""
        if logits is None:
            logits = self.last_added
        with tf.name_scope(self.get_name()):
            with tf.name_scope('eval'):
                self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.y, top_k), tf.float32),
                                               name='accuracy_top{}'.format(top_k))
                if self._logging[Utils.Logger.categories.EVAL]:
                    self.logger.add_summary('accuracy', tf.summary.scalar('accuracy', self.accuracy))
        return self.accuracy

    def _check_closed(self) -> bool:
        """Checks to see if the architecture is done being set up; throws an exception if it is"""
        if self._closed:
            raise EnvironmentError('Cannot add layers to this network after the architecture has been closed!')
        return True


class TextRNN(TextNet):
    """Class that represents a recurrent neural network used for text generation/classification"""

    def __init__(self, *args, **kwargs) -> None:
        """Creates an instance of a CNN

                Keyword Arguments:
                    learning_rate: The learning rate of this CNN (default 0.001)
                    logging: Tells if this TextCNN is to be logged to a file
                    verbose: Tells if updates to this neural network will be printed to the console
                    gm_time: Sets logging to Greenwich meantime rather than local time
                    log_file: The name of the log file for this TextRNN
                    encoding: The type of encoding for the characters used by this TextRNN
                    name: The name of this neural network
                """
        super().__init__(*args, **kwargs)


class TextCNN(TextNet):
    """Class to represent convolutional neural network for text generation/classification"""
    
    def __init__(self, *args, **kwargs) -> None:
        """Creates an instance of a CNN

        Keyword Arguments:
            learning_rate: The learning rate of this CNN (default 0.001)
            logging: Tells if this TextCNN is to be logged to a file
            verbose: Tells if updates to this neural network will be printed to the console
            gm_time: Sets logging to Greenwich meantime rather than local time
            log_file: The name of the log file for this TextCNN
            encoding: The type of encoding for the characters used by this TextCNN
            name: The name of this neural network
        """
        super().__init__(*args, **kwargs)


class TextGAN(TextNet):
    """Class to represent a Generative Adversarial Network specifically modeled for generating text"""

    def __init__(self, generator: TextNet, discriminator: TextNet, *args, **kwargs) -> None:
        """Creates an instance of a GAN

                Keyword Arguments:
                    learning_rate: The learning rate of this GAN (default 0.001)
                    logging: Tells if this TextGAN is to be logged to a file
                    verbose: Tells if updates to this neural network will be printed to the console
                    gm_time: Sets logging to Greenwich meantime rather than local time
                    log_file: The name of the log file for this TextGAN
                    encoding: The type of encoding for the characters used by this TextGAN
                    name: The name of this neural network
                """
        self.generator = generator
        self.discriminator = discriminator
        super().__init__(*args, **kwargs)
