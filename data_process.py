import scipy.io
from scipy.signal.windows import hann
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_signal(raw, lfp, spk):
    """
    :param raw: raw signal
    :param lfp: lfp filtered signal
    :param spk: spk filtered signal
    :return: None, only plots
    """
    samples_num = len(raw)
    time_in_sec_raw_spk = int(samples_num / 40000)
    time_in_sec_lfp = int(len(lfp) / 1000)

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(raw[:time_in_sec_raw_spk*60], 'r')  # row=0, col=0
    ax[0].set(xlabel='time (ms)',  ylabel="Voltage (V)")
    ax[0].set(title='raw')
    ax[1].plot(lfp[:time_in_sec_lfp*60], 'b')  # row=1, col=0
    ax[1].set(xlabel='time (ms)',  ylabel="Voltage (V)")
    ax[1].set(title='low')
    ax[2].plot(spk[:time_in_sec_raw_spk*600], 'g')  # row=0, col=1
    ax[2].set(xlabel='time (ms)',  ylabel="Voltage (V)")
    ax[2].set(title='high')
    plt.show()


def LFP_spectrum(LFP_signal, fs, motion):
    """
    :param LFP_signal: low frequency potentiation signal
    :param fs: sampling rate
    :param motion: move/still string
    :return: None, only plots
    """
    f, spectrum = signal.welch(LFP_signal, fs, window=hann(600),  nperseg=None, noverlap=600*0.25, nfft=8192, scaling='spectrum')
    spectrum = 10 * np.log10(spectrum/1e-6) # convert to DB

    f, t, spectrogram = signal.spectrogram(LFP_signal, fs,  window=hann(600), nperseg=None, nfft=8192,noverlap=600*0.25, scaling='spectrum')
    spectrogram = np.abs(spectrogram) # convert the power spectral density to power mean in mV
    spectrogram = 10 * np.log10(spectrogram / 1e-6)  # convert to DB

    fig, axs = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
    fig.suptitle(f'LFP Spectrum and spectrogram - {motion}', fontsize=18, c= 'c')
    # Plot spectrum:
    axs[0].semilogy(spectrum, f)
    axs[0].set_xlabel('Power Mean [Db]')
    axs[0].set_ylabel('frequency [Hz]')
    axs[0].set_title(f'Power Spectrum')
    axs[0].set_ylim([0, 30])
    axs[0].set_xlim([50, 100])
    # Plot spectrogram:
    im = axs[1].pcolormesh(t, f, spectrogram)
    axs[1].set_ylim([0, 30])
    axs[1].pcolormesh(t, f, spectrogram)
    axs[1].set_xlabel('time [s]')
    axs[1].set_title('Spectrogram')
    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label('Power Mean [Db]')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def firing_rate(spike_train, bin_size_sec, signal_len_sec):
    """
    :param spike_train: spk spike train
    :param bin_size_sec: desired sliding non overlapping window size
    :param signal_len_sec:
    :return: None, only plots
    """
    non_overlapping_windows_amount = round(signal_len_sec / bin_size_sec)
    # List of firing rates for each bin.
    bins_fr = []
    for i in range(non_overlapping_windows_amount):
        start = bin_size_sec * i
        # calculate firing rate for current bin
        fr_i = firing_rate_per_bin(spike_train, bin_size_sec, start) / bin_size_sec
        # append curr window firing rate to the list
        bins_fr.append(fr_i)

    yVec = []
    xVec = np.arange(0,signal_len_sec,1)
    for firing_rate_curr_bin in bins_fr:
        for j in range(bin_size_sec):
            yVec.append(firing_rate_curr_bin)
    plt.plot(xVec, yVec)
    plt.title(f"Firing rate, bin size of {bin_size_sec} seconds")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Firing rates (Hz)")
    plt.show()


def firing_rate_per_bin(spike_train, window_size, window_start_time):
    """
    :param spike_train: spk spike train
    :param window_size: sliding non overlapping window size
    :param window_start_time: t0 of the current bin
    :return: amount of spikes per one specific bin (current window position)
    """
    end = window_start_time + window_size
    spike_times_curr_window = []
    for spike_time in spike_train:
        if window_start_time <= spike_time < end:
            spike_times_curr_window.append(spike_time)
    return len(spike_times_curr_window)


def plot_firing_rates_different_windows(spike_train_spk_move, win_sizes, signal_len):
    """
    :param spike_train_spk_move: spike train
    :param win_sizes: array of different sliding non overlapping window sizes
    :param signal_len: signal duration
    :return: None - only plots
    The function plots firing rate histograms for each window size of the win_sizes array
    """
    for window in win_sizes:
        firing_rate(spike_train_spk_move, window, signal_len)


def main():
    # data loading
    raw_move = scipy.io.loadmat('data/Raw signal/CWB_Move.mat')['MoveWB'][0]
    low_move = scipy.io.loadmat('data/Low frequency/CLFP_Move.mat')['LFPMove'][0]
    high_move = scipy.io.loadmat('data/High frequency/CSPK_Move.mat')['MoveSPK'][0]
    plot_signal(raw_move, low_move, high_move)

    # spiking rate - SPK
    spike_train_spk_move = scipy.io.loadmat('data/spike trains/Train_move_spk.mat')['moves'][0]
    spike_train_spk_still = scipy.io.loadmat('data/spike trains/Train_still_spk.mat')['stills'][0]
    signal_len = 60 # seconds
    window_sizes = [1, 5, 10, 15, 20, 30]
    plot_firing_rates_different_windows(spike_train_spk_still, window_sizes, signal_len)

    # spectrum and spectrogram - LFP
    low_move = scipy.io.loadmat('data/Low frequency/CLFP_Move.mat')
    low_move_sampling_rate = low_move['sampling_freq'][0][0]
    low_move = low_move['LFPMove'][0]

    low_still = scipy.io.loadmat('data/Low frequency/CLFP_Still.mat')['LFPStill'][0]
    low_still_sampling_rate = scipy.io.loadmat('data/Low frequency/CLFP_Still.mat')['sampling_freq'][0][0]
    LFP_spectrum(low_move, low_move_sampling_rate, "move")
    LFP_spectrum(low_still, low_still_sampling_rate, "still")
    plt.clf()
    positions = spike_train_spk_move[:, np.newaxis]
    offsets = np.ones(len(positions))
    plt.eventplot(positions, linelengths=1, lineoffsets=offsets)
    plt.title('APs event plot - 100mV threshold')
    plt.xlabel('Time [seconds]')
    plt.show()

if __name__ == "__main__":
    main()
