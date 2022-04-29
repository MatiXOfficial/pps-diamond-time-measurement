import numpy as np


class CFD:
    """
    Constant Fraction Discriminator.
    """

    def __init__(self, n_baseline: int = 20, fraction: float = 0.3, choose_first: bool = True):
        """
        :param n_baseline: number of first values taken into account when calculating the baseline
        :param fraction: Peak fraction to return
        :param choose_first: if True: choose first peak
        """
        self.n_baseline = n_baseline
        self.fraction = fraction
        self.choose_first = choose_first

    def predict(self, Y: np.array, X: np.array = None, threshold: float = None, hysteresis: float = None, log: bool = False):
        """
        Find the timestamp
        :param Y: y axis data (ampl)
        :param X: x axis data (time). If None, set to an array from 0 to len(Y) - 1
        :param threshold: max - min threshold. if None: np.std(Y[:self.n_baseline]) * 6
        :param hysteresis: if None: threshold / 100
        :param log: if True, log messages are printed
        :return: timestamp
        """
        if X is None:
            X = np.arange(len(Y))
        if threshold is None:
            threshold = np.std(Y[:self.n_baseline]) * 6
        if hysteresis is None:
            hysteresis = threshold / 100

        samples = Y.astype(float)

        # if max - min < threshold there is no peak for sure
        if samples.max() - samples.min() < threshold:
            if log:
                print('max - min < threshold')
            return None

        # work only with positive and normalized signals
        samples -= np.mean(samples[:self.n_baseline])
        if abs(samples.max()) < abs(samples.min()):
            maximum = abs(samples.min())
            samples /= -samples.min()
        else:
            maximum = samples.max()
            samples /= samples.max()

        # threshold /= maximum
        hysteresis /= maximum

        above = False
        locked_for_hysteresis = False
        n_peaks = 0
        for i, v in enumerate(samples):
            if not above and not locked_for_hysteresis and v > self.fraction:
                first_crossing_index = i
                above = True
                locked_for_hysteresis = True
                if self.choose_first:
                    n_peaks += 1
                    above = False
                    break
            if above and locked_for_hysteresis and v > self.fraction + hysteresis:
                locked_for_hysteresis = False
            if above and not locked_for_hysteresis and v < self.fraction:
                n_peaks += 1
                above = False

        if n_peaks == 1:
            x1 = X[first_crossing_index]
            x2 = X[first_crossing_index - 1]
            v1 = samples[first_crossing_index]
            v2 = samples[first_crossing_index - 1]
            return x1 + (x2 - x1) * (self.fraction - v1) / (v2 - v1)
        else:
            if log:
                print('number of peaks:', n_peaks)
            return None
