"""Signal filtering: zero-phase Butterworth low-pass via scipy.signal.filtfilt.

When ``enabled`` is false in the config the signal passes through unchanged.
"""
import logging
import numpy as np

logger = logging.getLogger("sysid_pipeline")


def apply_filter(signal: np.ndarray, fs: float, cfg_filter: dict) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter to *signal*.

    Parameters
    ----------
    signal : (N,) or (N, nDoF) array
    fs : sampling frequency in Hz
    cfg_filter : filtering section of config, expects keys
        ``enabled`` (bool), ``cutoff_frequency_hz`` (float), ``filter_order`` (int)

    Returns
    -------
    Filtered signal, same shape as input.
    """
    if not cfg_filter.get("enabled", False):
        return signal

    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        logger.warning("scipy.signal not available -- returning unfiltered signal.")
        return signal

    cutoff = cfg_filter.get("cutoff_frequency_hz", 10.0)
    order = cfg_filter.get("filter_order", 4)
    nyquist = 0.5 * fs

    if cutoff >= nyquist:
        logger.warning("Cutoff %.1f Hz >= Nyquist %.1f Hz -- skipping filter.", cutoff, nyquist)
        return signal

    b, a = butter(order, cutoff / nyquist, btype="low")

    if signal.ndim == 1:
        return filtfilt(b, a, signal)

    out = np.empty_like(signal)
    for col in range(signal.shape[1]):
        out[:, col] = filtfilt(b, a, signal[:, col])

    logger.debug("Applied %d-order Butterworth low-pass at %.1f Hz (fs=%.1f Hz).",
                 order, cutoff, fs)
    return out
