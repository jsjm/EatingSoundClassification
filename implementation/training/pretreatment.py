"""
version: 1.0
creator: jsjm

References: 
Urban Sound Classification using Convolutional Neural Networks with Keras: Theory and Implementation
tutorial retrieved from: https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4

"""

import os
from glob import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path

def create_folders():
    os.mkdir('mel')
    os.mkdir('train')
    os.mkdir('validation')

def create_spectrogram(filename, name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72]) #param
    ax = fig.add_subplot(111) #param
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = "mel/" + name + '.png' 
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S

def run_transformation(dir, number_of_units):
    for i in range(number_of_units): 
        start = 0   
        unit = int(len(dir)/number_of_units)
        if i == number_of_units-1:
            start = int(end)
            end = None
        else:
            start = int(end)
            end = unit * (i + 1)    

        for file in dir[start:end]:
            filename,name = file,file.split('/')[-1].split('.')[0]
            create_spectrogram(filename,name)
            gc.collect()
            print ('done {0}%'.format(end/len(dir)))

if __name__ == '__main__':
    data_dir=glob("data/clips/*") # please change the path to your corresponding folders
    run_transformation(data_dir, 5) # please change the second parameter to larger numbers if you have a larger dataset
