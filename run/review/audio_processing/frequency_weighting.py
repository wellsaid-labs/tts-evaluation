""" This is a script to evaluate the current spectrogram weighting options in `lib.audio`.

Learn more about weighting here:
- https://en.wikipedia.org/wiki/Equal-loudness_contour
- Reference for A-Weighting and K-Weighting
  https://www.mathworks.com/help/audio/ref/weightingfilter-system-object.html
  https://en.wikipedia.org/wiki/A-weighting
- Reference for A-Weighting and ISO 226 Weighting
  https://en.wikipedia.org/wiki/A-weighting

Example:

    $ python -m run.review.audio_processing.frequency_weighting
"""
import numpy as np
from matplotlib import pyplot

from lib.audio import a_weighting, identity_weighting, iso226_weighting, k_weighting

if __name__ == "__main__":  # pragma: no cover
    sample_rate = 44100
    min_frequency = 10
    max_frequency = 21000
    num_points = 4096
    y_ticks = np.arange(-80, 20 + 1, 10)
    plot_kwargs = {
        "marker": ".",
        "linestyle": "solid",
        "markersize": 1,
    }
    frequencies = np.linspace(min_frequency, max_frequency, num_points, endpoint=True)

    pyplot.style.use("ggplot")  # type: ignore
    weighting = k_weighting(frequencies, sample_rate)
    pyplot.plot(frequencies, weighting, **plot_kwargs, label="K-Weighting")
    pyplot.plot(frequencies, a_weighting(frequencies), **plot_kwargs, label="A-Weighting")
    weighting = iso226_weighting(frequencies)
    pyplot.plot(frequencies, weighting, **plot_kwargs, label="ISO 226 Weighting")
    weighting = identity_weighting(frequencies)
    pyplot.plot(frequencies, weighting, **plot_kwargs, label="Identity Weighting")
    pyplot.legend()
    pyplot.yticks(y_ticks)
    pyplot.xlim(min_frequency, max_frequency)
    pyplot.ylabel("Magnitude (dB)")
    pyplot.xlabel("Frequency (Hz)")
    pyplot.xscale("log")
    pyplot.show()
