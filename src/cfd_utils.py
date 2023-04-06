from typing import List, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW

from src.cfd import CFD

TIME_STEP = 0.15625


def calculate_event_cfd(cfd: Union[CFD, List[CFD]], events: dict, i_event: int, shift: bool = True,
                        time_step: float = TIME_STEP, log: bool = False):
    """
    Calculate the cfd timestamps for all waveforms within an event
    :param cfd: A CFD instance for all channels or a list of CFD instances for each channel.
    :param events: Dict of lists containing event data
    :param i_event: Index of the considered event
    :param shift: if True: the timestamps will be shifted using their corresponding sample_t0 values
    :param time_step: time step between waveform samples
    :param log: log parameter for cfd
    :return: np array with a timestamp for each channel in the i_event-th event in events
    """
    event_ampl = events['sample_ampl'][i_event]
    n_channels = len(event_ampl)
    if type(cfd) is not list:
        cfd = [cfd for _ in range(n_channels)]

    event_cfd_timestamps = np.array([cfd[i].predict(event_ampl[i], log=log) for i in range(n_channels)])
    if shift:
        event_cfd_timestamps *= time_step

        event_t0 = events['sample_t0'][i_event]
        event_cfd_timestamps += event_t0

    return event_cfd_timestamps


def _gauss(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def _get_gauss_stats(y, x=None, return_all=False):
    if x is None:
        x = np.arange(len(y))

    # regular statistics
    weighted_stats = DescrStatsW(x, weights=y, ddof=0)
    mean_stat = weighted_stats.mean
    std_stat = weighted_stats.std

    # fitted gaussian statistics
    popt, pcov = curve_fit(_gauss, x, y, p0=[1, mean_stat, std_stat])
    gauss_a = popt[0]
    gauss_mean = popt[1]
    gauss_std = abs(popt[2])

    if return_all:
        return mean_stat, std_stat, gauss_mean, gauss_std, gauss_a, pcov
    else:
        return gauss_mean


def _diff_hist_stats(timestamps_diff, show, return_gauss_stats, n_bins, hist_range, hist_alpha, hist_label, plot_gauss,
                     return_pcov):
    hist_data = plt.hist(timestamps_diff, bins=n_bins, range=hist_range, alpha=hist_alpha, label=hist_label)

    # retrieve bins
    bins_x, bins_y = hist_data[1][:-1], hist_data[0]
    x_step = (bins_x[1] - bins_x[0]) / 2
    bins_x += x_step

    mean_stat, std_stat, gauss_mean, gauss_std, gauss_a, pcov = _get_gauss_stats(bins_y, bins_x, return_all=True)

    if plot_gauss:
        gauss_y = norm.pdf(bins_x, gauss_mean, gauss_std)
        gauss_y *= gauss_a / np.max(gauss_y)
        plt.plot(bins_x, gauss_y, 'r--', linewidth=2)

    if show:
        plt.show()

    retval = []
    if return_gauss_stats:
        retval += [gauss_mean, gauss_std]
    else:
        retval += [mean_stat, std_stat]

    if return_pcov:
        retval.append(pcov)

    return tuple(retval)


def find_diff_hist_stats(cfd: Union[CFD, List[CFD]], events: dict, show: bool = True, return_gauss_stats: bool = True,
                         n_bins: int = 100, hist_range: Tuple[float, float] = (-0.5, 1.5), hist_alpha: float = 1.,
                         hist_label: str = None, plot_gauss: bool = True, time_step: float = TIME_STEP,
                         return_pcov: bool = False, log: bool = False):
    """
    Find the mean and std of a histogram of differences between cfd timestamps in two channels
    :param cfd: A CFD instance for all channels or a list of CFD instances for each channel.
    :param events: Dict of lists containing event data
    :param show: If True: the histogram is shown (plt.show())
    :param return_gauss_stats: If True: the function returns the mean and std of a gaussian fitted to the histogram
    :param n_bins: Number of the histogram bins
    :param hist_range: Range of the histogram
    :param hist_alpha: Alpha of the plotted histogram
    :param hist_label: Label of the histogram
    :param plot_gauss: If True: a fitted Gaussian is plotted with the histogram
    :param time_step: time step between waveform samples
    :param return_pcov: If True, the covariance matrix of the Gaussian fit is returned
    :param log: If True, warnings are printed in the case of None returned by CFD
    :return: tuple: (mean, std) or (mean, std, cov) of the histogram
    """
    N = len(events['sample_t0'])

    # histogram
    timestamps = []
    nones = 0
    for i in range(N):
        try:
            timestamps.append(calculate_event_cfd(cfd, events, i, shift=True, time_step=time_step, log=log))
        except TypeError:
            nones += 1

    if log and nones > 0:
        print(f'Skipped {nones} Nones')

    timestamps = np.array(timestamps)
    timestamps_diff = timestamps[:, 1] - timestamps[:, 0]

    return _diff_hist_stats(timestamps_diff, show, return_gauss_stats, n_bins, hist_range, hist_alpha, hist_label,
                            plot_gauss, return_pcov)


def plot_diff_hist_stats(y_true, y_pred, show: bool = True, return_gauss_stats: bool = True, n_bins: int = 100,
                         hist_range: Tuple[float, float] = (-0.5, 1.5), hist_alpha: float = 1., hist_label: str = None,
                         plot_gauss: bool = True, return_pcov: bool = False):
    """
    Find the mean and std of a histogram of differences between y_true and y_pred timestamps
    :param y_true: Ground-truth timestamps
    :param y_pred: Predicted timestamps
    :param show: If True: the histogram is shown (plt.show())
    :param return_gauss_stats: If True: the function returns the mean and std of a gaussian fitted to the histogram
    :param n_bins: Number of the histogram bins
    :param hist_range: Range of the histogram
    :param hist_alpha: Alpha of the plotted histogram
    :param hist_label: Label of the histogram
    :param plot_gauss: If True: a fitted Gaussian is plotted with the histogram
    :param return_pcov: If True, the covariance matrix of the Gaussian fit is returned
    :return: tuple: (mean, std) or (mean, std, cov) of the histogram
    """

    # histogram
    timestamps_diff = y_pred - y_true

    plt.xlabel('time [156.25 ps]')
    return _diff_hist_stats(timestamps_diff, show, return_gauss_stats, n_bins, hist_range, hist_alpha, hist_label,
                            plot_gauss, return_pcov)


def plot_difference_hist(y_true, y_pred, channel, hist_range=(-2, 2), n_bins=100, xlabel=None, savefig=None, title=True,
                         ymax=None, fontsize=17, print_cov: bool = True, show: bool = True):
    plt.rc('font', size=fontsize)
    mu, std, pcov = plot_diff_hist_stats(y_true, y_pred, show=False, n_bins=n_bins, hist_range=hist_range,
                                         hist_label=None, plot_gauss=True, return_gauss_stats=True, return_pcov=True)

    if title is not None:
        plt.title(f'Diff histogram (channel={channel}), mean={mu:0.3f}, std={std:0.3f}')
    if xlabel is not None:
        plt.xlabel(xlabel)

    if ymax is not None:
        plt.ylim(top=ymax)

    plt.grid()
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)

    if show:
        plt.show()
    if print_cov:
        print('Covariance matrix of the Gaussian fit:')
        print(pcov)

    return std
