import logging
import os
import random, json
import time

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from masking import get_fake_batch, get_span_masked_feature
from utilis import InputExample, count_lines


class TextDatasetWriter:
    def __init__(self, dir_path, file_name):
        super().__init__()
        self.dir_path = dir_path
        self.file_name = file_name
        self.ext = ".txt"
        self.data_file = open(dir_path + "/" + file_name + self.ext, "w")

    def write_line(self, line, add_new_line=True):
        self.data_file.write(json.dumps(line))
        if add_new_line:
            self.data_file.write("\n")

    def close_writer(self):
        self.data_file.close()


class BinaryIndexDatasetWriter:
    def __init__(self, dir_path, file_name):
        super().__init__()
        self.dir_path = dir_path
        self.file_name = file_name
        self.ext = ".bin"
        self.data_file = open(dir_path + "/" + file_name + self.ext, "wb")
        self.dtype = np.int32
        self.byte_size = 4
        self.offsets = [0]
        self.array_sizes = []

    def write_line(self, tokens):
        tok_array = np.array(tokens, dtype=self.dtype)
        chars = self.data_file.write(tok_array)
        self.offsets.append(self.offsets[-1] + chars / self.byte_size)
        # print(self.array_sizes,"vjhfvjhg", tok_array, type(tok_array))
        self.array_sizes.append(tok_array.size)

    def close_writer(self):
        self.data_file.close()
        idx_info = {
            "offsets": self.offsets,
            "sizes": self.array_sizes,
            "len": len(self.array_sizes),
        }
        with open(
            self.dir_path + "/" + self.file_name + "_indexing.json", "w"
        ) as idx_f:
            json.dump(idx_info, idx_f)
            # print("index written sucess")
