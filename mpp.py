import os
from collections import defaultdict
from glob import glob
from typing import Dict, List, Union

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


class MarkovModel:
    """Represents a Markov model of given text input"""

    def __init__(self, regex: str='', k: int=12) -> None:
        """Initializes a Markov model to a given set of directories filled with text files

        Keyword Arguments:
            regex - The glob pattern to grab directories of text files to use (default cwd)
            k - The order of this Markov model (default 12)
        """
        self._subs: Dict[str, Markov] = {}
        self._texts = set()
        self.k = k
        self.__add__(regex)
        self.starts = []

    def __add__(self, regex: str) -> None:
        """Adds text files to this Markov model. Accepts globs to directory locations. Use '' for cwd"""
        for f in MarkovModel.get_files(regex):
            with open(f, 'r') as io:
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
                    self._subs[sub].add_suffix(text[i + self.k + 1])

    @staticmethod
    def get_files(regex: str) -> List[str]:
        files = []
        for path in glob(regex if regex != '' else os.getcwd()):
            # Skip all non-directories
            if not os.path.isdir(path):
                continue
            # List all the files in the directory being used
            for f in os.listdir(path):
                # Skip previously read texts and nested directories (only use plain files)
                if os.path.isdir(f):
                    continue
                files.append(f)
        return files

    def get_subs(self) -> List[str]:
        """Returns a list of all substrings in this Markov model"""
        return list(self._subs.keys())

    def get_suffixes(self, substring: str, with_freq=False) -> Union[List[chr], Dict[chr, float]]:
        """Returns either a list of all possible suffixes of a substring in this Markov model, or a dict containing
        those suffixes mapped to their frequencies"""
        if substring not in self._subs:
            MarkovModel._missing(substring)
        suffs = self._subs[substring].suffixes()
        if with_freq:
            total = sum(suffs[c] for c in suffs)
            return {c: suffs[c] / total for c in suffs}
        else:
            return list(suffs.keys())

    def random_suffix(self, substring: str) -> chr:
        if substring not in self._subs:
            MarkovModel._missing(substring)
        return self._subs[substring].random()

    def get_start(self) -> str:
        """Returns a random starting string (the first k characters in any of the files used)"""
        return np.random.choice(self.starts)

    @staticmethod
    def _missing(substring: str) -> None:
        """Raises error for missing substring in this Markov model"""
        raise KeyError('{} not in this Markov model'.format(substring))
