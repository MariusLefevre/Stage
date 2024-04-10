#Test#
import matplotlib.pyplot as plt
import librosa
import scipy as sp
from scipy import signal
from scipy import special
import libfmp.b
import numpy as np
import csv
import py_midicsv as pm


def to_csv(timestamps):
    with open('test.csv', 'w', newline='') as csvfile:
        fieldnames = ['0','1','2','3','4','5','6']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)

        csvfile.write("0, 1, Header, 1, 2, 44100\n")
        csvfile.write("1, 0, Start_track\n")
        csvfile.write("1, 0, Title_t, \"Close Encounters\"\n")
        csvfile.write("1, 0, Text_t, \"Sample for MIDIcsv Distribution\"\n")
        csvfile.write("1, 0, Copyright_t, \"This file is in the public domain\"\n")
        csvfile.write("1, 0, Time_signature, 4, 2, 24, 8\n")
        csvfile.write("1, 0, Tempo, 500000\n")
        csvfile.write("1, 0, End_track\n")
        csvfile.write("2, 0, Start_track\n")
        csvfile.write("2, 0, Instrument_name_t, \"Piano\"\n")
        csvfile.write("2, 0, Program_c, 1, 1\n")
        for i in timestamps:
            csvfile.write("1, ")
            csvfile.write(str(int(i/20*441000)))
            csvfile.write(", Note_on_c,1,70,100\n")
            csvfile.write("1, ")
            csvfile.write(str(int(i/20*441000)+100))
            csvfile.write(", Note_off_c,1,70,100\n")
        csvfile.write("2, 441000, End_track\n")
        csvfile.write("0, 0, End_of_file")

    
    return 



def compute_novelty_energy(x, Fs=1, N=2048, H=128, gamma=10.0, norm=True):
    """Compute energy-based novelty function

    Notebook: C6/C6S1_NoveltyEnergy.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 2048)
        H (int): Hop size (Default value = 128)
        gamma (float): Parameter for logarithmic compression (Default value = 10.0)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_energy (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    # x_power = x**2
    w = signal.hann(N)
    Fs_feature = Fs / H
    energy_local = np.convolve(x**2, w**2, 'same')
    energy_local = energy_local[::H]
    if gamma is not None:
        energy_local = np.log(1 + gamma * energy_local)
    energy_local_diff = np.diff(energy_local)
    energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
    novelty_energy = np.copy(energy_local_diff)
    novelty_energy[energy_local_diff < 0] = 0
    if norm:
        max_value = max(novelty_energy)
        if max_value > 0:
            novelty_energy = novelty_energy / max_value
    return novelty_energy

def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100.0, M=10, norm=True):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

def fromNoveltyToTimestamps(novelty,audiosize,threshold):
    """Compute timestamps from novelty function using threshold

    Args:
        novelty (np.ndarray): input function
        audiosize (int): Size of the entry audio file
        threshold (float): Treshold under which the novelty spike wont be considered

    Returns:
        timeStamps (list): timestamps in samples
    """
    x=len(novelty)
    timeStamps=[]
    neighbor = -100
    window = 10
    for i in range(x):
        if(novelty[i]>threshold):
            if(i>neighbor+window):
                timeStamps.append((i/x)*audiosize/sr)
            neighbor=i
    return timeStamps

def subdivide_beats(beats):
    subdividedbeats=beats.copy()
    i=0
    while  i < len(subdividedbeats)-1:
        subdividedbeats.insert(i+1,(subdividedbeats[i]+subdividedbeats[i+1])/2)
        i+=2
    return subdividedbeats

def find_closest_beat(timestamp,beats):
    closest_beat=100000
    for beat in beats:
        if (np.abs(timestamp-beat)>np.abs(timestamp-closest_beat)):
            closest_beat=beat
    return closest_beat
    

def quantize_timestamps(timestamps,beats,strengh):
    subdividedbeats=subdivide_beats(beats)
    quantized_timestamps=[]
    for timestamp in timestamps:
        closest_beat=find_closest_beat(timestamp,subdividedbeats)
        if(closest_beat<strengh):
            quantized_timestamps.append(closest_beat)
        else:
            quantized_timestamps.append(timestamp)

    return quantized_timestamps


label_keys={'linewidth': 1, 'linestyle': ':', 'color': 'b'}
label_keys2={'linewidth': 1, 'linestyle': ':', 'color': 'k'}

y, sr = librosa.load('test2.mp3')
plp = librosa.beat.plp(y=y,sr=sr,tempo_min=50,tempo_max=150)
tempo, beats = librosa.beat.beat_track(y=y, units='time', trim=True,tightness=10)
onsets = librosa.onset.onset_detect(y=y,hop_length=100, units='time')
#fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1 ,2 ]}, figsize=(6, 2))
novelty1=compute_novelty_energy(y, Fs=sr)
novelty2,fs=compute_novelty_spectrum(y, Fs=sr)
ax1 = plt.subplot(311)
wave=librosa.display.waveshow(y=y,ax=ax1,offset=0 ,sr=sr,color="blue") 
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
libfmp.b.plot_signal(novelty1, Fs=sr,ax=ax2, color='k', title='Novelty function (original)')
#ax4 = plt.subplot(313)
libfmp.b.plot_signal(novelty2, Fs=sr,ax=ax2, color='green', title='Novelty function (original)')

timestamps=fromNoveltyToTimestamps(novelty2,len(y),0.05 )
"""
ax3 = plt.subplot(315)
libfmp.b.plot_signal(novelty3, Fs=sr,ax=ax3, color='k', title='Novelty function (original)')
ax3 = plt.subplot(316)
libfmp.b.plot_signal(novelty4, Fs=sr,ax=ax3, color='k', title='Novelty function (original)')
"""
beats=beats.tolist()

for i in reversed(range(len(beats))):
    if(i>len(beats)):
        break
    if( i % 2 == 1): 
        beats.pop(i)

libfmp.b.plot_annotation_line(quantize_timestamps(timestamps,beats,1000),ax3,time_min=0)
libfmp.b.plot_annotation_line(beats,ax3,colors='prism',dpi=128,time_min=0)
libfmp.b.plot_annotation_line(onsets,ax1,label_keys=label_keys2,dpi=500,time_min=0)



#plt.plot()
print(len(novelty1),len(novelty2),len(y),tempo/2)

plt.show()

to_csv(quantize_timestamps(timestamps,beats,1000))

csv_string = ""

with open("test.csv", "r") as f:
    csv_string=f.readlines(100000)
# Parse the CSV output of the previous command back into a MIDI file
midi_object = pm.csv_to_midi(csv_string)

# Save the parsed MIDI file to disk
with open("test.mid", "wb") as output_file:
    midi_writer = pm.FileWriter(output_file)
    midi_writer.write(midi_object)
