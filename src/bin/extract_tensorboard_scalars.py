""" Script to extract tensorboard scalars.

Script inspired by:
https://github.com/tensorflow/tensorboard/issues/706
https://gist.github.com/wchargin/31eee50b9aaebf387b380f70054575c5
"""
import argparse
import csv
import errno
import os
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
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    return multiplexer


def export_scalars(multiplexer, run, tag, filepath, write_headers=True):
    if os.path.isfile(filepath):
        os.remove(filepath)

    data = extract_scalars(multiplexer, run, tag)
    with open(filepath, 'w') as outfile:
        writer = csv.writer(outfile)
        if write_headers:
            writer.writerow(('wall_time', 'step', 'value'))
        for row in data:
            writer.writerow(row)


NON_ALPHABETIC = re.compile('[^A-Za-z0-9_]')


def munge_filename(name):
    """ Remove characters that might not be safe in a filename. """
    return NON_ALPHABETIC.sub('_', name)


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(directory)):
            raise


def main(run, tags, output_dir='/tmp/csv_output'):
    mkdir_p(output_dir)

    tf.logging.info('Loading data...')
    multiplexer = create_multiplexer(ROOT_PATH)
    assert os.path.isdir(os.path.join(
        ROOT_PATH, run)), 'Run directory must exist %s' % os.path.join(ROOT_PATH, run)
    for tag_name in tags:
        output_filename = '%s___%s.csv' % (munge_filename(run), munge_filename(tag_name))
        output_filepath = os.path.join(output_dir, output_filename)
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
