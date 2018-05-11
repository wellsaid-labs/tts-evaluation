import os

from src.loss import plot_loss
from src.loss import Loss



def test_plot_loss():
    filename = 'tests/_test_data/loss.png'
    plot_loss([[4, 3, 2, 1], [3, 2, 1, 0]], ['Train', 'Valid'], filename)

    assert os.path.isfile(filename)
    # Clean up
    os.remove(filename)

def test_loss():
