from praatio import textgrid
import librosa
from librosa import display
from librosa.effects import preemphasis
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random


#Spectrify (2022) - Arian Shamei, Yadong Liu, Vancouver BC


# Spectrify is an interface for praat.io and Librosa that allows rapid large scale spectrographic data generation from force-aligned material.
#The purpose of spectrify is to segment speech into consistent chunks based on word level or phoneme level alignment in praat textgrids, while avoiding large silences or disruptions.
# This allows the generation of multiple spectrograms from an audiofile while ensuring complete units of analysis are present within each spectrogram, and that the spectrogram only contains the target utterance.


#Input parameter to Spectrify() consists of a vector including the target directory (containing paired audio & textgrids),as well as spectrographic parameters
#Example:
#           input =[directory,"fmin","fmax","nmels","hop_length","nfft"]
#           The order of arguments must be followed exactly and all arguments must be specified.

condition = "control"
directory = "D:\\UBC\\alzheimer\\english_data\\"+condition+"\\Cookies\\female"
if not os.path.exists(directory+"\\mfcc"):
    os.makedirs(directory+"\\mfcc")

input =[directory,100,4000,64,64,2048]
#load textgrid


#take textgrid and identify large silences (>100ms), plan 1s spectrograms excluding any marked silence.
def planner(input,filename):
    tg = textgrid.openTextgrid(filename, False)
    entryList_phone = tg.tierDict["sentence - phones"].entryList
    entryList_word = tg.tierDict["sentence - words"].entryList
    for item in entryList_phone:
        if item[-1] == "sp":
            entryList_word.append(item)
    phraser(sorted(entryList_word, key=lambda tup: tup[0]),input,filename)


def phraser(word_withsp,input,filename):
    phrases = []
    phrase_duration = 0
    word_count = 0
    for i in range(len(word_withsp)):
        phrase = []
        duration = word_withsp[i][1] - word_withsp[i][0]
        phrase_duration += duration

        word_count += 1
        if phrase_duration >= 1:
            for word in word_withsp[i - word_count + 1:i + 1]:
                phrase.append(word)
            phrases.append(phrase)
            phrase_duration = 0
            word_count = 0
    indexer(phrases,input,filename)

def indexer(phrases,input,filename):
    phrase_index = set()
    for i in range(len(phrases)):
        for word in phrases[i]:
            if word[-1] == "sp" and word[1] - word[0] > 0.1:
                print(word)
                phrase_index.add(i)
    indexed = [j for i, j in enumerate(phrases) if i not in phrase_index]
    phrases_time = []
    for phrase in indexed:
        phrase_time = (phrase[0][0], phrase[-1][1])
        phrases_time.append(phrase_time)
    spectrify(phrases_time,input,filename)


def spectrify(phrases_time,input,filename):
    i=0
    filename=str(filename[:-9])+".wav"
    speaker = filename.split("\\")[-1][:-4]
    plt.figure(figsize=(1, 1))
    for stamp in phrases_time:
        try:
            i=i+1
            length = stamp[1] - stamp[0]
            y, sr = librosa.load(filename, offset=stamp[0], duration=length)

            S = librosa.feature.mfcc(y=y, sr=sr, n_mels=input[3], n_fft = input[5], hop_length=input[4],
                                           fmin=input[1],fmax=input[2])
            S_db = librosa.power_to_db(S)
            librosa.display.specshow(S_db,sr=sr,fmin=300,
                                     fmax=150)
            plt.tight_layout()
            plt.savefig(directory+"\\mfcc\\" + speaker + "-"+str(i)+".png")
            plt.clf()
        except:
            print("missing file for "+str(filename))




def Spectrify(input):
    for filename in os.listdir(input[0]):
        if filename.endswith(".TextGrid"):
            planner(input,input[0]+"\\"+filename)

Spectrify(input)


