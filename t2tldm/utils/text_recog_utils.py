
import logging
import os
import time

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor, nn

class CharsetMapper(object):
    """A simple class to map ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.
    """

    def __init__(self,
                 filename='',
                 max_length=30,
                 null_char=u'\u2591'):

        """Creates a lookup table.

        Args:
          filename: Path to charset file which maps characters to ids.
          max_sequence_length: The max length of ids and string.
          null_char: A unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.
        """
        self.null_char = null_char
        self.max_length = max_length

        self.label_to_char = self._read_charset(filename)
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)

    def _read_charset(self, filename):
        """Reads a charset definition from a tab separated text file.

        Args:
          filename: a path to the charset file.

        Returns:
          a dictionary with keys equal to character codes and values - unicode
          characters.
        """
        import re
        pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        self.null_label = 0
        charset[self.null_label] = self.null_char
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                m = pattern.match(line)
                assert m, f'Incorrect charset file. line #{i}: {line}'
                label = int(m.group(1)) + 1
                char = m.group(2)
                charset[label] = char
        return charset
    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, '')

    def get_text(self, labels, length=None, padding=True, trim=False):
        """ Returns a string corresponding to a sequence of character ids.
        """
        length = length if length else self.max_length
        labels = [l.item() if isinstance(l, Tensor) else int(l) for l in labels]
        if padding:
            labels = labels + [self.null_label] * (length-len(labels))
        text = ''.join([self.label_to_char[label] for label in labels])
        if trim: text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """ Returns the labels of the corresponding text.
        """
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length

        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return '0123456789'

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                valid_chars.append(c)
        return ''.join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)
class Config(object):

    def __init__(self, config_path, host=True):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_')
                else:
                    if k == 'phase':
                        assert v in ['train', 'test']
                    if k == 'stage':
                        assert v in ['pretrain-vision', 'pretrain-language',
                                     'train-semi-super', 'train-super']
                    self.__setattr__(f'{prefix}{k}', v)

        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        with open('data/template.yaml') as file:
            default_config_dict = yaml.load(file, Loader=yaml.FullLoader)
        __dict2attr(default_config_dict)
        __dict2attr(config_dict)
        self.global_workdir = os.path.join(self.global_workdir, self.global_name)

    def __getattr__(self, item):
        attr = self.__dict__.get(item)
        if attr is None:
            attr = dict()
            prefix = f'{item}_'
            for k, v in self.__dict__.items():
                if k.startswith(prefix):
                    n = k.replace(prefix, '')
                    attr[n] = v
            return attr if len(attr) > 0 else None
        else:
            return attr

    def __repr__(self):
        str = 'ModelConfig(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str