import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW

from src.cfd import CFD

TIME_STEP = 0.15625


def calculate_event_cfd(cfd: CFD, events: dict, i_event: int, shift: bool = True, time_step: float = TIME_STEP,
                        log: bool = False):
    """
    Calculate the cfd timestamps for all waveforms within an event
    :param cfd: A CFD instance
    :param events: Dict of lists containing event data
    :param i_event: Index of the considered event
    :param shift: if True: the timestamps will be shifted using their corresponding sample_t0 values
    :param time_step: time step between waveform samples
    :param log: log parameter for cfd
    :return: np array with a timestamp for each channel in the i_event-th event in events
    """
    event_ampl = events['sample_ampl'][i_event]
    n_events = len(event_ampl)

    event_cfd_timestamps = np.array([cfd.predict(event_ampl[i], log=log) for i in range(n_events)])
    if shift:
        event_cfd_timestamps *= time_step

        event_t0 = events['sample_t0'][i_event]
        event_cfd_timestamps += event_t0

    return event_cfd_timestamps


def _gauss(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def find_diff_hist_stats(cfd: CFD, events: dict, show: bool = True, return_gauss_stats: bool = True,
                         hist_range: tuple[float, float] = (-0.5, 1.5), hist_alpha: float = 1., hist_label: str = None,
                         plot_gauss: bool = True, time_step: float = TIME_STEP):
    """
    Find the mean and std of a histogram of differences between cfd timestamps in two channels
    :param cfd: A CFD instance
    :param events: Dict of lists containing event data
    :param show: If True: the histogram is shown (plt.show())
    :param return_gauss_stats: If True: the function returns the mean and std of a gaussian fitted to the histogram
    :param hist_range: Range of the histogram
    :param hist_alpha: Alpha of the plotted histogram
    :param hist_label: Label of the histogram
    :param plot_gauss: If True: a fitted Gaussian is plotted with the histogram
    :param time_step: time step between waveform samples
    :return: tuple: (mean, std) of the histogram
    """
    N = len(events['sample_t0'])

    # histogram
    timestamps = np.array(
        [calculate_event_cfd(cfd, events, i, shift=True, time_step=time_step, log=True) for i in range(N)])
    timestamps_diff = timestamps[:, 1] - timestamps[:, 0]
    hist_data = plt.hist(timestamps_diff, bins=100, range=hist_range, alpha=hist_alpha, label=hist_label)

    # retrieve bins
    bins_x, bins_y = hist_data[1][:-1], hist_data[0]
    x_step = (bins_x[1] - bins_x[0]) / 2
    bins_x += x_step

    # regular statistics
    weighted_stats = DescrStatsW(bins_x, weights=bins_y, ddof=0)
    mean_stat = weighted_stats.mean
    std_stat = weighted_stats.std

    # fitted gaussian statistics
    popt, _ = curve_fit(_gauss, bins_x, bins_y, p0=[1, mean_stat, std_stat])
    gauss_mean = popt[1]
    gauss_std = abs(popt[2])

    if plot_gauss:
        gauss_y = norm.pdf(bins_x, gauss_mean, gauss_std)
        gauss_y *= np.max(bins_y) / np.max(gauss_y)
        plt.plot(bins_x, gauss_y, 'r--', linewidth=2)

    if show:
        plt.show()

    if return_gauss_stats:
        return gauss_mean, gauss_std
    else:
        return mean_stat, std_stat
