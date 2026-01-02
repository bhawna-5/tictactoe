import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=False)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename_h5 = filename.split(".")[0] + ".h5"
        
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath_h5 = os.path.join(folder, filename_h5)
        filepath_original = os.path.join(folder, filename)
        
        # Try .h5 file first, then try original filename (might be .h5 with wrong extension)
        if os.path.exists(filepath_h5):
            filepath = filepath_h5
        elif os.path.exists(filepath_original):
            # If original file exists but .h5 doesn't, check if it's actually an H5 file
            # Keras load_weights needs .h5 extension to recognize HDF5 format
            import h5py
            try:
                # Try to open as H5 to verify format
                with h5py.File(filepath_original, 'r') as f:
                    # File is H5 format, create temp copy with .h5 extension
                    temp_h5_path = filepath_original + '.h5'
                    if not os.path.exists(temp_h5_path):
                        shutil.copy2(filepath_original, temp_h5_path)
                    filepath = temp_h5_path
            except:
                # Not an H5 file, try loading directly (might be TensorFlow format)
                filepath = filepath_original
        else:
            raise FileNotFoundError("No model in path '{}' or '{}'".format(filepath_h5, filepath_original))
        
        # Load the weights
        self.nnet.model.load_weights(filepath)
        
        # Clean up temp file if we created one
        if filepath.endswith('.h5') and filepath != filepath_h5 and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass  # Ignore cleanup errors
