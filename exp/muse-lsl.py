# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""
# TODO:: saves the beta power in textfile or something
# TODO:: laplace filter for the channels?

import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt  # Module used for plotting
import numpy as np  # Module that simplifies computations on matrices

# visual stuff
import pygame
from pygame.locals import *
from pylsl import (  # Module to receive EEG data
    StreamInlet,
    proc_clocksync,
    proc_dejitter,
    resolve_byprop,
)

import utils  # Our own utility functions

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 3

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.5

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# Fz = 1, CP5 = 10
INDEX_CHANNEL = [23]

save_power = True

## pygame init and stuff
pygame.init()

fps = 15
timeout = 1 / fps

fpsClock = pygame.time.Clock()

screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_w, screen_h = pygame.display.get_surface().get_size()

background_color = [127, 127, 127]
text_fontsize = 75
text_color = [64, 64, 64]
char_fontsize = 100
char_color = [0, 0, 0]
shade_of_blue = [139, 255]
init_shade = shade_of_blue[1]
blue = (0, 0, init_shade)

# init heights
init_height = 250
weight = 300
init_width = 100
max_height = 400
min_height = 50


def draw_rect(width, height, col):
    rect = pygame.Rect(0, 0, width, height)
    offset_height = 400
    new_height = offset_height - height / 2
    rect.center = (screen_w / 2, screen_h / 2 + new_height)
    screen.fill([0, 0, 0])  # Fill the entire screen with black
    pygame.draw.rect(screen, col, rect)


## main program

if __name__ == "__main__":

    """1. CONNECT TO EEG STREAM"""

    # Search for active LSL streams
    print("Looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=2)
    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream.")

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(
        streams[0], max_chunklen=12, processing_flags=proc_clocksync | proc_dejitter
    )
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()
    print(info)
    print(description)
    # print("info ", info)

    fs = int(info.nominal_srate())
    print("fs sampling rate ", fs)

    #############################

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(INDEX_CHANNEL)))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    band_buffer = np.zeros((n_win_test, 1))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print("Press Ctrl-C in the console to break the while loop.")
    draw_rect(init_width, init_height, blue)

    ts_arr = list()
    power_8hz_arr = list()
    power_high_alpha_arr = list()

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            ## for pygame to quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=timeout, max_samples=int(SHIFT_LENGTH * fs)
            )

            # eeg_data, timestamp = inlet.pull_sample()
            # print(timestamp)
            ts_arr.extend(timestamp)

            # print(np.array(eeg_data).shape)
            np_eeg_data = np.array(eeg_data)
            if np_eeg_data.ndim > 1:
                # Only keep the channel we're interested in

                ch_data = np_eeg_data[:, INDEX_CHANNEL]
                # print(ch_data.shape)
                # Update EEG buffer with the new data
                eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True, filter_state=filter_state
                )

                """ 3.2 COMPUTE BAND POWERS """
                # Get newest samples from the buffer
                data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
                # Compute band powers
                # band_powers = utils.compute_band_powers(data_epoch, fs)
                mean_8hz, mean_high_alpha = utils.compute_band_powers(data_epoch, fs)
                # feature = [mean_high_alpha/mean_8hz]
                feature = [mean_high_alpha]

                band_buffer, _ = utils.update_buffer(band_buffer, np.asarray(feature))
                # Compute the average band powers for all epochs in buffer
                # This helps to smooth out noise
                smooth_band_powers = np.mean(band_buffer, axis=0)

                ## append to array to save to file later
                power_8hz_arr.append(mean_8hz[0])
                power_high_alpha_arr.append(mean_high_alpha[0])

                """ 3.3 COMPUTE NEUROFEEDBACK METRICS """

                height = init_height + smooth_band_powers[0] * weight
                # height = init_height + feature[0][0] * weight
                draw_rect(init_width, height, blue)

            # make it run in the same sampling frequnecy of the EEG.
            pygame.display.update()
            fpsClock.tick(fps)

    except KeyboardInterrupt:
        # print(ts_arr)
        # print(np.diff(np.array(ts_arr)))

        # save power list into text file
        if save_power:

            filename = datetime.today().strftime("%Y%m%d%H%M%S") + "_power.txt"
            with open(filename, "w") as f:
                f.write("8Hz,10-12Hz\n")
                # print(len(power_8hz_arr))
                # print(len(power_high_alpha_arr))
                for i in range(len(power_8hz_arr)):
                    # for power in power_8hz_arr:
                    f.write("%s,%s\n" % (power_8hz_arr[i], power_high_alpha_arr[i]))
                print("Saved power file ", filename)
        print("Closing!")
