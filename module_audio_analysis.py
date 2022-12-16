# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import pandas as pd
import librosa
import librosa.display
#import IPython.display as ipd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy import signal 
from scipy.fft import rfftfreq, rfft
import scipy.stats as stats
from sklearn.manifold import TSNE
import random
import os
import fnmatch
import time




class audio:

    def __init__(self):
        a = 1


    def load_aud_wnoise(self, y, sr, noise_start, noise_end,mode):
        """Segment the audio data and remove noise
        Argv (inputs):
            y = input audio signal
            sr = sampling rate of the system
            start, end = segment legth

        returns (output):
            noise removed audio segment of desired start , end legth

        """
        #print(y.shape, sr)
        tt = len(y) / sr
        # enter time in sec
        start_t = int(noise_start * sr)
        end_t = int(noise_end * sr)

        noise_seg = y[start_t: end_t]
        #aud_seg = y

        avg_noise = np.mean(noise_seg)
        y_net = y - avg_noise

        y_net_cutsound = y_net [int(0): start_t]
        y_net_noise = y_net [start_t : end_t]
        #print("avg noise {}".format(avg_noise))
        if mode == "cut" :
            return y_net_cutsound
        if mode == "noise" :
            return y_net_noise

       


    def visual_l(self, y_net, sr):
        """Takes in a audio signal and outputs time series data"""
        plt.figure()
        librosa.display.waveshow(y=y_net, sr=sr)
        # librosa.display.waveplot(y= aud_seg  ,sr = sr)
        plt.xlabel("Time (seconds) ")
        plt.ylabel("Amplitude")
        plt.show()

        return 0

    def visual_ps(self, y_net, sr):
        """Takes in a audio signal and outputs power spectrum of the data """
        # finding the stft of the signal segment
        D_short = librosa.stft(y_net, hop_length=64, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D_short), ref=np.max)

        # code for power spectrogram
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_db, sr=sr, hop_length=64, x_axis='time', y_axis='log', ax=ax)
        ax.set(title='Power Spectrogram {}'.format)
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        return 0

    def visual_ls(self, y_net, sr, xvalue):
        """Takes in a audio signal and outputs power spectrum of the data """
        # finding the stft of the signal segment
        D_short = librosa.stft(y_net, hop_length=64, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D_short), ref=np.max)

        # linear spectrogram
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_db, sr=sr, hop_length=64, x_axis='time', y_axis='linear', ax=ax)
        ax.set(title='Spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.ylim([xvalue, 8000])
        # plt.ylim([xvalue,8000])

        return 0

    def visual(self, y_net, sr):
        """Plots 3 plots together for quick visualization"""
        print("Function Visual")
        fig, ax = plt.subplots(nrows=3, sharex=True)

        librosa.display.waveshow(y=y_net, sr=sr, ax=ax[0], label="Raw signal")

        # finding the stft of the signal segment
        D_short = librosa.stft(y_net, hop_length=64, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D_short), ref=np.max)

        # code for power spectrogram

        img = librosa.display.specshow(S_db,sr=sr, hop_length=64, x_axis="time", y_axis='log', ax=ax[1])
        # ax[1].set(title='Power Spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        # linear spectrogram
        img = librosa.display.specshow(S_db,sr= sr, hop_length=64, x_axis='time', y_axis='linear', ax=ax[2])
        # ax[2].set(title='P_9')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        # plt.ylim([0,8000])
        plt.ylim([0, 8000])

        #plt.title("{}".title)
        plt.show()

        return 1


    # Filters

 

    def notch_filter(self,y, sr, notch_freq):
        """Removes a particular frequency band from the audio signal"""
        samp_freq = sr  # Sample frequency (Hz)
        # notch_freq = 5000  # Frequency to be removed from signal (Hz)
        quality_factor = 30.0  # Quality factor
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
        freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
        y_notch = signal.filtfilt(b_notch, a_notch, y)
        print("frequency band removed {}".format(notch_freq))
        return y_notch

    def butter_bandstop_filter(self,y, lowcut, highcut, fs, order):
        """Removes a particular frequency band from the audio signal"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        i, u = signal.butter(order, [low, high], btype='bandstop')
        y = signal.lfilter(i, u, y)

        print("frequency band removed {}, {}".format(lowcut, highcut))
        return y

    def butter_highpass_filter(self,data, sr, hpass_freq):
        """Removes frequency below a frequency value from the audio signal"""
        sos = signal.butter(10, hpass_freq, "hp", fs=sr, output="sos")
        filtered = signal.sosfilt(sos, data)
        print("frequency band removed below {}".format(hpass_freq))
        return filtered

    def calc_mfcc(self,y, sr, n_mfcc):
        """calculates MFCC co-efficients of given order"""
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)

        return mfcc

    def get_pca(self,features, n):
        """Performs PCA on a given set of features, where n is the order of operation"""

        pca = PCA(n_components=n)
        transformed = pca.fit(features).transform(features)
        scaler = MinMaxScaler()
        scaler.fit(transformed)

        return scaler.transform(transformed)



    def test_visual(self,contact_mic_left, contact_mic_right, noncontact_mic,sr,cut_idx):
        """ Takes the input audio files and plots the results for quick visualization
        """

        print("Function Visual cut idx {}".format(cut_idx))
        fig, ax = plt.subplots(nrows=2,ncols=2 , sharex=True, constrained_layout = True)

        # finding the stft of the contact mic left signal segment
        D_short_contact_left = librosa.stft(contact_mic_left, hop_length=64, win_length=512)
        S_db_contact_left = librosa.amplitude_to_db(np.abs(D_short_contact_left), ref=np.max)
        img = librosa.display.specshow(S_db_contact_left,sr=sr, hop_length=64, x_axis="time", y_axis='linear', ax=ax[0][0])
        #fig.colorbar(img, ax=ax, format="%+2.f dB")

        # finding the stft of the contact mic right signal segment
        D_short_contact_right = librosa.stft(contact_mic_right, hop_length=64, win_length=512)
        S_db_contact_right = librosa.amplitude_to_db(np.abs(D_short_contact_right), ref=np.max)
        img = librosa.display.specshow(S_db_contact_right,sr=sr, hop_length=64, x_axis="time", y_axis='linear', ax=ax[0][1])
       # fig.colorbar(img, ax=ax, format="%+2.f dB")

        # finding the stft of the noncontact mic left signal segment
        D_short_noncontact_left = librosa.stft(noncontact_mic[0], hop_length=64, win_length=512)
        S_db_noncontact_left = librosa.amplitude_to_db(np.abs(D_short_noncontact_left), ref=np.max)
        img = librosa.display.specshow(S_db_noncontact_left,sr=sr, hop_length=64, x_axis="time", y_axis='linear', ax=ax[1][0])
        #fig.colorbar(img, ax=ax, format="%+2.f dB")

        # finding the stft of the contact mic right signal segment
        D_short_noncontact_right = librosa.stft(noncontact_mic[1], hop_length=64, win_length=512)
        S_db_noncontact_right = librosa.amplitude_to_db(np.abs(D_short_noncontact_right), ref=np.max)
        img = librosa.display.specshow(S_db_noncontact_right,sr=sr, hop_length=64, x_axis="time", y_axis='linear', ax=ax[1][1])
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        #plt.title("{}".title)
        plt.show()

    def get_tsne(self,features, n,random_state,perplexity):
        """Performs t-SNE on a given set of features, where n is the order of operation"""
        n_iter = 5000
        tsne = TSNE(n_components=n, verbose=0, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_results = tsne.fit_transform(features)

        #print("tsne_shape {}".format(tsne_results.shape))

        return tsne_results

    def tsne_from_mfcc(self, mfcc,n_components,perplexity,file_name):
        """ Takes MFCC array and plots tsne for all the values
        """
        range = np.array(mfcc[:,0,0].shape).squeeze()

       # range = np.array(mfcc[:,0].shape).squeeze()
        print("range {}".format(range))
        i= 0
        tsne = np.zeros((range,20,2))
        fig = plt.figure(figsize = (5,5))
        # ax = fig.add_subplot(111, projection="3d")
                
        ax = plt.axes()
        while i < range:

            rand = random.randint(1,1000)
            print("index = {}".format(i))
            tsne[i,:]= self.get_tsne(mfcc[0],n_components,rand,perplexity)
            # 3D plot
            # ax.scatter(tsne[i,:].T[0],tsne[i,:].T[1],tsne[i,:].T[2],s = 30)
            # ax.set_title( "perplexity {}".format(perplexity))
            # plt.savefig(file_name + "perplexity_{}".format(perplexity))

            # 2D plot
            ax= plt.scatter(tsne[i,:].T[0],tsne[i,:].T[1])#,tsne[i,:].T[2],s = 30)
            plt.title( "perplexity {}".format(perplexity))
            plt.savefig(file_name + "perplexity_{}".format(perplexity))

            i = i+ 1


    def dataset_analysis(self,path_microphone,cut_idx):
        """ loads files from a folder into a dataframe and calculates corresponding MFCC coefficients
        """
        #os.chdir("/Users/rp/Library/CloudStorage/Box-Box/Home Folder rp247/Private/1. Documents_220111/Duke/1. BTL/1b. Photoablation_Acoustics/PAA_code_data/data/1.Dataset")
        
        idx = 0
        file_counter = cut_idx
        no_of_cuts = 18
        end_cut_idx = cut_idx + no_of_cuts

        array_value = [431,130,301] 
        # choose 431 for 5 sec sound for MFCC 20 
        # choose 140 for 1.5 sec sound for MFCC 20
        # choose 301 for 3.5 sec sound for MFCC 20

        mfcc_contact_left = np.zeros((no_of_cuts,20,array_value[1]))
        mfcc_contact_right = np.zeros((no_of_cuts,20,array_value[1]))
        mfcc_noncontact_left = np.zeros((no_of_cuts,20,array_value[1]))
        mfcc_noncontact_right = np.zeros((no_of_cuts,20,array_value[1]))

        print("cut idx {} end_cut_idx {}".format(cut_idx,end_cut_idx))

        while file_counter < end_cut_idx:
            
            print("cut_idx {}, idx {}".format(cut_idx,idx))   
            # load the files
            contact_mic_left , sr = librosa.load(path_microphone + "/contact_mic_left_" + str(cut_idx+idx)+ ".wav", sr = 44100)
            contact_mic_right , sr = librosa.load(path_microphone + "/contact_mic_right_" + str(cut_idx +idx)+ ".wav", sr = 44100) 
            noncontact_mic , sr = librosa.load(path_microphone + "/noncontact_mic_" + str(cut_idx +idx)+ ".wav", sr = 44100, mono = False)  # mono = False because of 2 channel recording

            # substract noise from the audio files
            contact_mic_left = ad.load_aud_wnoise(contact_mic_left,sr,noise_start=1.5,noise_end = 5,mode ="cut")
            contact_mic_right = ad.load_aud_wnoise(contact_mic_right,sr,noise_start=1.5,noise_end = 5,mode ="cut")
            # noncontact_mic[0] = ad.load_aud_wnoise(noncontact_mic[0],sr,noise_start=1.5,noise_end = 5)
            # noncontact_mic[1] = ad.load_aud_wnoise(noncontact_mic[1],sr,noise_start=1.5,noise_end = 5)
            noncontact_mic_left = ad.load_aud_wnoise(noncontact_mic[0],sr,noise_start=1.5,noise_end = 5,mode ="cut")
            noncontact_mic_right = ad.load_aud_wnoise(noncontact_mic[1],sr,noise_start=1.5,noise_end = 5,mode ="cut")
            noncontact_mic = [noncontact_mic_left,noncontact_mic_right]

            # find MFCC coefficients : audio features
            mfcc_contact_left[idx, :] = self.calc_mfcc(contact_mic_left,sr,n_mfcc=20)
            mfcc_contact_right[idx, :] = self.calc_mfcc(contact_mic_right,sr,n_mfcc=20)
            mfcc_noncontact_left[idx, :] = self.calc_mfcc(noncontact_mic[0],sr,n_mfcc=20)
            mfcc_noncontact_right[idx, :] = self.calc_mfcc(noncontact_mic[1],sr,n_mfcc=20) 

            # print("Shape of contact_left mfcc {}".format(mfcc_contact_left.shape))
            # print("Shape of contact_right mfcc {}".format(mfcc_contact_right.shape))
            # print("Shape of noncontact_left mfcc {}".format(mfcc_noncontact_left.shape))
            # print("Shape of noncontact_left mfcc {}".format(mfcc_noncontact_right.shape)) 

            idx = idx+1
            file_counter = file_counter + 1
        #print("MFCC_contact_left {}".format(mfcc_contact_left[0].shape))

        """ After finding the MFCC , time to calculate tsne
        """

        #print("range shape {}".format(mfcc_contact_left[:,0,0].shape))
        print("---------------------------")

       
        for p in range(5,25,5):
            print("perplexity {}".format(p))
            self.tsne_from_mfcc(mfcc_contact_left,n_components=2,perplexity=p,file_name = "contact_mic_left")
            self.tsne_from_mfcc(mfcc_contact_right,n_components=2,perplexity=p,file_name = "contact_mic_right")
            self.tsne_from_mfcc(mfcc_noncontact_left,n_components=2,perplexity=p,file_name = "noncontact_mic_left")
            self.tsne_from_mfcc(mfcc_noncontact_right,n_components=2,perplexity=p,file_name = "noncontact_mic_right")

            


    def data_fft(self,contact_mic_left,contact_mic_right,noncontact_mic_left,noncontact_mic_right,sr,cut_idx):
        """ Generates FFT plots for audio files in a folder
        """

        cut_idx = int(cut_idx)
        audio_files = [contact_mic_left,contact_mic_right,noncontact_mic_left,noncontact_mic_right]
        audio_filename = ["contact_mic_left","contact_mic_right","noncontact_mic_left","noncontact_mic_right"]
        counter = 0

        audio_fft = np.zeros((4,33076))
        while counter < 4 : 
            
            yf = rfft(audio_files[counter])
            N = len(audio_files[counter]) 
            xf = rfftfreq(N, 1/sr)
            audio_fft[counter,:] = yf.squeeze()

            fig = plt.figure(figsize = (5,4))
            ax = plt.axes()
            ax.xaxis.set_major_locator(MultipleLocator(10000))
            ax.xaxis.set_minor_locator(MultipleLocator(2500))
            ax.plot(xf,np.abs(yf),color = "#11151C" )
            
            plt.title("{}, cut {}".format(audio_filename[counter],cut_idx))
            ax.set_xlabel("Freq (Hz)")
            ax.set_ylabel("FFT Amplitude |X(freq)|")
           plt.savefig("{}, cut {}".format(audio_filename[counter],cut_idx))
            print(audio_filename[counter])
            counter = counter + 1

                    
        return audio_fft
        

    def data_fft_difference(self,microphone_audio,sr,cut_idx):
        """ Find a FFT of a audio signal and noise 
            Return: Plot of difference of FFT

            NOTE: Function does not work at the moment
        """
        mic_cut = ad.load_aud_wnoise(microphone_audio,sr,noise_start=1.5,noise_end = 3,mode ="cut")
        mic_noise = ad.load_aud_wnoise(microphone_audio,sr,noise_start=1.5,noise_end = 3,mode ="noise")

        fft_cut = rfft(mic_cut)
        fft_noise = rfft(mic_noise)
        N = len(mic_cut) 
        xf = rfftfreq(N, 1/sr)

        fft_diff = fft_cut - fft_noise

        fig = plt.figure(figsize = (5,3))
        ax = plt.axes()
        ax.xaxis.set_major_locator(MultipleLocator(10000))
        ax.xaxis.set_minor_locator(MultipleLocator(2500))
        ax.plot(xf,np.abs(fft_diff),color = "#11151C" )
        
        plt.title("{}, cut {}".format(microphone_audio,cut_idx))
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("FFT Amplitude |X(freq)|")

    def analysis_fft (self, cut_idx, path_microphone):
        """ Calculate FFT as feature set and perform PCA
            NOTE: Function not working
        """

        pca_components = 50
        idx = 0
        file_counter = cut_idx
        no_of_cuts = 18
        end_cut_idx = cut_idx + no_of_cuts

        # shape of FFT array determined from FFT function for a 1.5 sec audio at 44000 Hz sampling rate.
        fft_contact_left = np.zeros((no_of_cuts,33076))
        fft_contact_right = np.zeros((no_of_cuts,33076))
        fft_noncontact_left = np.zeros((no_of_cuts,33076))
        fft_noncontact_right = np.zeros((no_of_cuts,33076))

        # shape of PCA array determined from FFT function for a 1.5 sec audio at 44000 Hz sampling rate.
        pca_contact_left = np.zeros((no_of_cuts,pca_components))
        pca_contact_right = np.zeros((no_of_cuts,pca_components))
        pca_noncontact_left = np.zeros((no_of_cuts,pca_components))
        pca_noncontact_right = np.zeros((no_of_cuts,pca_components))

        

        while file_counter < end_cut_idx :
        
            # # load the files
            contact_mic_left , sr = librosa.load(path_microphone + "/contact_mic_left_" + str(file_counter)+ ".wav", sr = 44100)
            contact_mic_right , sr = librosa.load(path_microphone + "/contact_mic_right_" + str(file_counter)+ ".wav", sr = 44100) 
            noncontact_mic , sr = librosa.load(path_microphone + "/noncontact_mic_" + str(file_counter)+ ".wav", sr = 44100, mono = False)  # mono = False because of 2 channel recording

            # Remove noise from the audio files
            contact_mic_left = self.load_aud_wnoise(contact_mic_left,sr,noise_start=1.5,noise_end = 5,mode ="cut")
            contact_mic_right = self.load_aud_wnoise(contact_mic_right,sr,noise_start=1.5,noise_end = 5,mode ="cut")
            noncontact_mic_left = self.load_aud_wnoise(noncontact_mic[0],sr,noise_start=1.5,noise_end = 5,mode ="cut")
            noncontact_mic_right = self.load_aud_wnoise(noncontact_mic[1],sr,noise_start=1.5,noise_end = 5,mode ="cut")
            noncontact_mic = [noncontact_mic_left,noncontact_mic_right]

           # print("shape of audio signal {}".format(contact_mic_left.shape))
            # find FFT coefficients : audio features
            fft_features = ad.data_fft(contact_mic_left,contact_mic_right,noncontact_mic_left,noncontact_mic_right,sr,file_counter)
            fft_contact_left[idx , : ] = fft_features[0]
            fft_contact_right[idx , : ] = fft_features[1]
            fft_noncontact_left[idx , : ] = fft_features[2]
            fft_noncontact_right[idx , : ] = fft_features[3]
            
            # pca = self.get_pca(fft_features[0].reshape(-1,1),2)
            # print("shape of pca {}".format(pca))

            # # reduce dimensions of FFT features using PCA 
            # pca_contact_left[counter , : ] = self.get_pca(fft_contact_left[counter],pca_components)
            # pca_contact_right[counter , : ] = self.get_pca(fft_contact_right[counter],pca_components)
            # pca_noncontact_left[counter , : ] = self.get_pca(fft_noncontact_left[counter],pca_components)
            # pca_noncontact_right[counter , : ] = self.get_pca(fft_noncontact_right[counter],pca_components)
             

            idx = idx+1
            file_counter = file_counter + 1
           
           
        fft = [fft_contact_left,fft_contact_right,fft_noncontact_left,fft_noncontact_right]
        #fft = np.transpose(fft)
        fft = np.array(fft)
        print("Shape of fft {}".format(fft.shape))

        self.tsne_from_mfcc(fft[0],n_components=2,perplexity = 10,file_name="contact_left")
        
        # print("shape of pca contact left {}".format(pca_contact_left.shape))
        plt.clf()
      

            




#%%

if __name__ == "__main__" :

    ad = audio()
    
   
    dir = r"C:/Users/rapra/Box/Home Folder rp247/Private/1. Documents_220111/Duke/1. BTL/1b. Photoablation_Acoustics/PAA_code_data/data/1.Dataset"
    os.chdir(dir)
    path_microphone = "./221126_P2"
    cut_idx = 180

    """THIS SECTION IS FOR TESTING FOR ROBOT LEARNING CLASS
    """

    #ad.dataset_analysis(path_microphone,cut_idx)

    ad.analysis_fft(cut_idx,path_microphone)

    


  

 
    
# %%
