""" Script inspired with:
https://github.com/tensorflow/tensorboard/issues/706
https://gist.github.com/wchargin/31eee50b9aaebf387b380f70054575c5
"""
import csv
import errno
import os
import re

import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer
from src.utils import ROOT_PATH

# Control downsampling: how many scalar data do we keep for each run/tag
# combination?
SIZE_GUIDANCE = {'scalars': 100000000}


def extract_scalars(multiplexer, run, tag):
    """Extract tabular data from the scalars at a given run and tag.

  The result is a list of 3-tuples (wall_time, step, value).
  """
    tensor_events = multiplexer.Tensors(run, tag)
    return [(event.wall_time, event.step, tf.make_ndarray(event.tensor_proto).item())
            for event in tensor_events]


def create_multiplexer(logdir):
    multiplexer = event_multiplexer.EventMultiplexer(tensor_size_guidance=SIZE_GUIDANCE)
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
    """Remove characters that might not be safe in a filename."""
    return NON_ALPHABETIC.sub('_', name)


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(directory)):
            raise


def main():
    run_names = ('data/tensorboard/train',)
    tag_names = ('coarse/loss/step', 'parameter_norm/step', 'max_grad_norm/step')

    logdir = ROOT_PATH
    output_dir = '/tmp/csv_output'
    mkdir_p(output_dir)

    print("Loading data...")
    multiplexer = create_multiplexer(logdir)
    for run_name in run_names:
        assert os.path.isdir(os.path.join(logdir, run_name))
        for tag_name in tag_names:
            output_filename = '%s___%s.csv' % (munge_filename(run_name), munge_filename(tag_name))
            output_filepath = os.path.join(output_dir, output_filename)
            print("Exporting (run=%r, tag=%r) to %r..." % (run_name, tag_name, output_filepath))
            export_scalars(multiplexer, run_name, tag_name, output_filepath)
    print("Done.")


if __name__ == '__main__':
    main()
