import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, List, Union
import time

import numpy as np


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

    def __init__(self, regex: str=None, k: int=12) -> None:
        """Initializes a Markov chain to a given set of directories filled with text files

        Keyword Arguments:
            regex - The glob pattern to grab directories of text files to use (default cwd)
            k - The order of this Markov chain (default 12)
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
        possible = re.compile(r'^' + path + r'(\([0-9]+\))?' + r'$')
        # Get a list of all files/dirs in the cwd depending on whether or not is_dir is true
        files = [f for f in glob('*') if possible.match(f) is not None and (is_dir == os.path.isdir(f))]
        last = max(int(f[-2]) if f.endswith(')') else 0 for f in files) if len(files) > 0 else -1
        return path + ('' if last == -1 else '({})'.format(str(last + 1)))


class MarkovNN:
    """Class that represents a recurrent neural network used for text generation"""
    _unnamed = 0

    def __init__(self,
                 learning_rate=0.001,
                 neurons=20,
                 logging: bool=False, verbose: bool=False, gm_time: bool=False, log_file: str=None,
                 name=None) -> None:
        """Creates an instance of an RNN

        Keyword Arguments:
            learning_rate - The learning rate of this RNN (default 0.001)
            neurons - The number of neurons for the input layer (default 20)
            logging - Tells if this MarkovNN is to be logged to a file
            verbose - Tells if updates to this neural network will be printed to the console
            gm_time - Sets logging to Greenwich meantime rather than local time
            log_file - The name of the log file for this MarkovNN
            name - The name of this neural network
        """
        self._verbose: bool = verbose
        self._learning_rate: float = float(learning_rate)
        self._name: str = str(name) if name is not None else None
        if name is None:
            self._name = 'unnamed_MarkovRNN-' + str(MarkovNN._unnamed)
            MarkovNN._unnamed += 1
        self._log_file: str = Utils.safe_path('log/' + log_file if log_file is not None else self._name + '-log')
        self._logging: bool = logging
        self._neurons: int = int(neurons)
        self._log_time = time.gmtime if gm_time else time.localtime
        self.log('Initialize ' + str(self))

    def get_name(self) -> str:
        """Returns the generic name of this RNN"""
        return self._name

    def log(self, message: str=None) -> None:
        """Logs a message or just the time if no message is provided. Covers both printing and file logging"""
        out = time.strftime('[%a, %d %b %Y %H:%M:%S] ', self._log_time()) + message if message is not None else ''
        if self._verbose:
            print(out)
        if self._logging:
            with open(self._log_file, 'a+', encoding='utf-8') as log:
                log.write(out + '\n')

    def __str__(self) -> str:
        """Returns a string representation of this MarkovNN object (includes pertinent information)"""
        return 'mpp.MarkovNN(name={}, neurons={}, learning_rate={}'.format(
            self._name, self._neurons, self._learning_rate)
