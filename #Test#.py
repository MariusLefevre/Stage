#Test#

import matplotlib.pyplot as plt
import librosa
import scipy as sp
from scipy import signal
import libfmp.b
import numpy as np
import py_midicsv as pm
import easygui

duration=30

def to_secs(liste : list):
    out=[]
    size=len(liste)
    for element in liste:
        out.append(element/size*duration)
    return out

def calculateTsPerBeat(tempo,lengh,duration):
    return lengh/duration*60/tempo

def to_csv(timestamps,beats,lengh,tempo):
    with open('test.csv', 'w', newline='') as csvfile:

        csvfile.write("0, 1, Header, 1, 2,"+ str(int(calculateTsPerBeat(tempo,lengh,duration)))+"\n")
        csvfile.write("1, 0, Start_track\n")
        csvfile.write("1, 0, Title_t, \"Close Encounters\"\n")
        csvfile.write("1, 0, Text_t, \"Sample for MIDIcsv Distribution\"\n")
        csvfile.write("1, 0, Copyright_t, \"This file is in the public domain\"\n")
        csvfile.write("1, 0, Time_signature, 4, 2, 24, 8\n")
        csvfile.write("1, 0, Tempo,"+str(int(60000000/tempo)    )+"\n")
        csvfile.write("1, 0, End_track\n")
        csvfile.write("2, 0, Start_track\n")
        csvfile.write("2, 0, Instrument_name_t, \"Piano\"\n")
        csvfile.write("2, 0, Program_c, 1, 1\n")

        for i in timestamps:
            csvfile.write("1, ")
            csvfile.write(str(int(i)))
            csvfile.write(", Note_on_c,1,70,100\n")
            csvfile.write("1, ")
            csvfile.write(str(int(i+500)))
            csvfile.write(", Note_off_c,1,70,100\n")
        csvfile.write("2, "+str(lengh)+", End_track\n")
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
    window = 5
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

def avg_env(enveloppe):
    summ=0
    for i in range(len(enveloppe)):
        summ+=enveloppe[i]
    return summ/len(enveloppe)
    
def cap_env(enveloppe,threshold):
    for i in range(len(enveloppe)):
        if(enveloppe[i]<threshold):
            enveloppe[i] = threshold
    return enveloppe

def find_closest_beat(timestamp,beats,lengh):
    closest_beat=100000
    for beat in beats:
        if (np.abs(timestamp-beat)<np.abs(closest_beat)):
            closest_beat=timestamp-beat
    
    return closest_beat
    
def quantize_timestamps(timestamps,beats,strengh,tempo,lengh):
    correctedbeats=[]
    i=0
    tspb=calculateTsPerBeat(tempo,lengh,duration)
    while i*tspb<lengh:
        correctedbeats.append(i*tspb)
        i+=1
    subdividedbeats=subdivide_beats(subdivide_beats(correctedbeats))
    #print(subdividedbeats)
    quantized_timestamps=[]

    for timestamp in timestamps:
        quantized_timestamps.append(int((timestamp-timestamps[0])/duration*lengh))



    for timestamp in quantized_timestamps:
        closest_beat=find_closest_beat(timestamp,subdividedbeats,lengh)
        if(np.abs(closest_beat<strengh)):
            for j in range(quantized_timestamps.index(timestamp),len(quantized_timestamps)):
                #print(str(j)+":"+str(timestamps[j])+"--->"+str(timestamps[j]-closest_beat))
                quantized_timestamps[j]=quantized_timestamps[j]-closest_beat

        #else:
         #   quantized_timestamps.append(timestamp)
    
    return quantized_timestamps

label_keys={'linewidth': 1, 'linestyle': ':', 'color': 'b'}
label_keys2={'linewidth': 1, 'linestyle': ':', 'color': 'k'}

y, sr = librosa.load(easygui.fileopenbox(),duration=duration)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

plp = librosa.beat.plp(onset_envelope=onset_env,tempo_min=30,tempo_max=300,win_length=300)

tempo, beats = librosa.beat.beat_track(y=y, units='time', trim=True,tightness=10)
if(tempo>130):tempo = tempo/2
elif(tempo<30):tempo = tempo*2  
onsets = librosa.onset.onset_detect(y=y,hop_length=100, units='time')

novelty1=compute_novelty_energy(y, Fs=sr)
novelty2,fs=compute_novelty_spectrum(y)
#timestamps=fromNoveltyToTimestamps(librosa.util.normalize(onset_env),len(y),0.08)
beats=beats.tolist()
D = np.abs(librosa.stft(y))
times = librosa.times_like(onset_env)
beats_plp = librosa.util.localmax(plp)
half_env=onset_env-avg_env(onset_env)
lissed_env=cap_env(half_env,0)
half_env = signal.savgol_filter(half_env,10,3)
timestamps2=fromNoveltyToTimestamps(librosa.util.normalize(lissed_env),len(y),0.3)

ax1 = plt.subplot(311)
ax3 = plt.subplot(312,sharex=ax1)
ax2 = plt.subplot(313,sharex=ax1)
wave=librosa.display.waveshow(y=y,ax=ax1,offset=0 ,sr=sr,color="blue") 
ax3.plot(times,librosa.util.normalize(onset_env), alpha=0.8,label='onsetEnvelope?')
ax3.plot(times,librosa.util.normalize(half_env), alpha=0.8,label='onsetEnvelope?')
ax1.plot(times,beats_plp,color="red")
ax2.plot(times,librosa.util.normalize(lissed_env))

libfmp.b.plot_annotation_line(timestamps2*duration,ax1,label_keys=label_keys2,dpi=500,time_min=0, time_max=duration)
#libfmp.b.plot_annotation_line(timestamps*duration,ax1,label_keys=label_keys2,dpi=500,time_min=0, time_max=duration)
#plt.plot()

print(len(novelty1),len(novelty2),len(y),tempo,len(plp))
plt.show()

to_csv(quantize_timestamps(timestamps2,beats,500,tempo,len(y)),beats,len(y),tempo)
csv_string = ""

with open("test.csv", "r") as f:
    csv_string=f.readlines(100000)
midi_object = pm.csv_to_midi(csv_string)

with open("test.mid", "wb") as output_file:
    midi_writer = pm.FileWriter(output_file)
    midi_writer.write(midi_object)
