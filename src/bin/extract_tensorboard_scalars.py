""" Script to extract tensorboard scalars.

Script inspired by:
https://github.com/tensorflow/tensorboard/issues/706
https://gist.github.com/wchargin/31eee50b9aaebf387b380f70054575c5

Example:

    python3 -m src.bin.extract_tensorboard_scalars \
            --tags coarse/loss/step \
            --run experiments/experiment/tb/train
"""
from pathlib import Path

import argparse
import csv
import re
import sys

from src.utils import ROOT_PATH
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def extract_scalars(multiplexer, run, tag):
    """Extract tabular data from the scalars at a given run and tag.

    The result is a list of 3-tuples (wall_time, step, value).
    """
    tensor_events = multiplexer.Tensors(run, tag)
    return [(event.wall_time, event.step, tf.make_ndarray(event.tensor_proto).item())
            for event in tensor_events]


def create_multiplexer(logdir):
    multiplexer = event_multiplexer.EventMultiplexer(tensor_size_guidance={'scalars': sys.maxsize})
    multiplexer.AddRunsFromDirectory(str(logdir))
    multiplexer.Reload()
    return multiplexer


def export_scalars(multiplexer, run, tag, filepath, write_headers=True):
    if filepath.is_file():
        filepath.unlink()

    data = extract_scalars(multiplexer, run, tag)
    with filepath.open(mode='w') as outfile:
        writer = csv.writer(outfile)
        if write_headers:
            writer.writerow(('wall_time', 'step', 'value'))
        for row in data:
            writer.writerow(row)


NON_ALPHABETIC = re.compile('[^A-Za-z0-9_]')


def munge_filename(name):
    """ Remove characters that might not be safe in a filename. """
    return NON_ALPHABETIC.sub('_', name)


def main(run, tags, output_dir='/tmp/csv_output'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tf.logging.info('Loading data...')
    multiplexer = create_multiplexer(ROOT_PATH)
    assert Path(run).is_dir(), 'Run directory must exist %s' % run
    for tag_name in tags:
        output_filename = munge_filename('%s___%s' % (run, tag_name)) + '.csv'
        output_filepath = output_dir / output_filename
        tf.logging.info('Exporting (run=%r, tag=%r) to %r...', run, tag_name, output_filepath)
        export_scalars(multiplexer, run, tag_name, output_filepath)
    tf.logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tags', nargs='+', help='Scalar tag names to extract', required=True)
    parser.add_argument('-r', '--run', type=str, required=True, help='Path to run to extract')
    args = parser.parse_args()
    main(**vars(args))
