""" This is a script to demonstrate the current spectrogram weighting options in `src.audio`.

Learn more about weighting here:
https://en.wikipedia.org/wiki/Equal-loudness_contour
https://en.wikipedia.org/wiki/A-weighting
"""
from matplotlib import pyplot

import numpy as np

from src.audio import a_weighting
from src.audio import k_weighting
from src.audio import iso226_weighting

sample_rate = 44100
min_frequency = 10
max_frequency = 21000
num_points = 4096
y_ticks = np.arange(-80, 20 + 1, 10)
plot_kwargs = {
    'marker': '.',
    'linestyle': 'solid',
    'markersize': 1,
}

frequencies = np.linspace(min_frequency, max_frequency, num_points, endpoint=True)

# Learn more:
# - Reference for A-Weighting and K-Weighting
#   https://www.mathworks.com/help/audio/ref/weightingfilter-system-object.html
# - Reference for A-Weighting and ISO 226 Weighting
#   https://en.wikipedia.org/wiki/A-weighting
pyplot.style.use('ggplot')
pyplot.plot(frequencies, k_weighting(frequencies, sample_rate), **plot_kwargs, label='K-Weighting')
pyplot.plot(frequencies, a_weighting(frequencies), **plot_kwargs, label='A-Weighting')
pyplot.plot(frequencies, iso226_weighting(frequencies), **plot_kwargs, label='ISO 226 Weighting')
pyplot.yticks(y_ticks)
pyplot.xlim(min_frequency, max_frequency)
pyplot.ylabel('Magnitude (dB)')
pyplot.xlabel('Frequency (Hz)')
pyplot.xscale('log')
pyplot.show()
