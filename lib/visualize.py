import io
import logging
import os
import time
import types
import typing
from contextlib import contextmanager

import numpy as np
import torch
from hparams import HParam, configurable
from third_party import LazyLoader

import lib

if typing.TYPE_CHECKING:  # pragma: no cover
    import comet_ml
    import matplotlib
    from librosa import display as librosa_display
    from matplotlib import pyplot
else:
    comet_ml = LazyLoader("comet_ml", globals(), "comet_ml")
    librosa_display = LazyLoader("librosa_display", globals(), "librosa.display")
    matplotlib = LazyLoader("matplotlib", globals(), "matplotlib")
    pyplot = LazyLoader("pyplot", globals(), "matplotlib.pyplot")

logger = logging.getLogger(__name__)

# fmt: off
turbo_colormap_data = [
    [0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024],
    [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314],
    [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096],
    [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373],
    [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142],
    [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406],
    [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162],
    [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412],
    [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156],
    [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393],
    [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123],
    [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347],
    [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064],
    [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275],
    [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930],
    [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773],
    [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835],
    [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202],
    [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959],
    [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190],
    [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981],
    [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416],
    [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580],
    [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559],
    [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437],
    [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299],
    [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230],
    [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316],
    [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599],
    [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627],
    [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308],
    [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713],
    [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914],
    [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981],
    [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987],
    [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002],
    [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098],
    [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345],
    [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816],
    [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581],
    [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712],
    [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280],
    [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [0.64362, 0.98999, 0.23356],
    [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960],
    [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032],
    [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504],
    [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311],
    [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386],
    [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663],
    [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074],
    [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555],
    [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038],
    [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456],
    [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744],
    [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836],
    [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663],
    [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161],
    [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358],
    [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348],
    [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165],
    [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842],
    [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412],
    [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910],
    [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367],
    [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817],
    [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294],
    [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831],
    [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461],
    [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218],
    [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134],
    [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243],
    [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521],
    [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875],
    [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305],
    [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809],
    [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387],
    [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041],
    [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769],
    [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571],
    [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449],
    [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401],
    [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427],
    [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529],
    [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705],
    [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955],
    [0.47960, 0.01583, 0.01055],
]
# fmt: on

try:
    pyplot.style.use("ggplot")
    matplotlib.cm.register_cmap("turbo", cmap=matplotlib.colors.ListedColormap(turbo_colormap_data))
except NameError:
    logger.info("Ignoring optional `matplotlib` dependency.")


def plot_alignments(alignment: typing.Union[torch.Tensor, np.ndarray]) -> matplotlib.figure.Figure:
    """Plot an alignment of two sequences.

    Args:
        alignment (numpy.ndarray or torch.Tensor [decoded_sequence, encoded_sequence]): Alignment
            weights between two sequences.
    """
    alignment = alignment.detach().cpu().numpy() if torch.is_tensor(alignment) else alignment
    alignment = np.transpose(alignment)
    figure, axis = pyplot.subplots()
    im = axis.imshow(
        alignment,
        cmap="turbo",
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(im, ax=axis, orientation="horizontal")
    pyplot.xlabel("Decoded Sequence")
    pyplot.ylabel("Encoded Sequence")
    pyplot.close(figure)
    return figure


def plot_logits(logits: typing.Union[torch.Tensor, np.ndarray]) -> matplotlib.figure.Figure:
    """Given a time-series of logits, plot a line graph.

    Args:
        logits (numpy.array or torch.Tensor [sequence_length])
    """
    logits = logits if torch.is_tensor(logits) else torch.tensor(logits)
    logits = torch.sigmoid(logits)
    logits = logits.detach().cpu().numpy()
    figure = pyplot.figure()
    pyplot.plot(list(range(len(logits))), logits, marker=".", linestyle="solid")
    pyplot.ylabel("Probability")
    pyplot.xlabel("Timestep")
    pyplot.close(figure)
    return figure


@configurable
def plot_waveform(
    signal: typing.Union[torch.Tensor, np.ndarray], sample_rate: int = HParam()
) -> matplotlib.figure.Figure:
    """Plot the amplitude envelope of a waveform.

    Args:
        signal (torch.Tensor or numpy.array [signal_length]): Signal to plot as a waveform.
        sample_rate: The number of samples or points per second.
    """
    signal = signal.detach().cpu().numpy() if torch.is_tensor(signal) else signal
    figure = pyplot.figure()
    librosa_display.waveplot(signal, sr=sample_rate)
    pyplot.ylabel("Amplitude")
    pyplot.xlabel("Time")
    pyplot.ylim(-1.0, 1.0)
    # Learn more: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    pyplot.close(figure)
    return figure


@configurable
def plot_mel_spectrogram(
    spectrogram: typing.Union[torch.Tensor, np.ndarray],
    lower_hertz: typing.Optional[int] = HParam(),
    upper_hertz: typing.Optional[int] = HParam(),
    y_axis: str = "mel",
    **kwargs,
) -> matplotlib.figure.Figure:
    """Plot a mel spectrogram.

    Args:
        spectrogram (numpy.array or torch.FloatTensor [num_frames, num_mel_bins])
        lower_hertz: Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz: The desired top edge of the highest frequency band.
        **kwargs: Any additional keyword arguments are passed onto `visualize.plot_spectrogram`.
    """
    return plot_spectrogram(
        spectrogram, fmin=lower_hertz, fmax=upper_hertz, y_axis=y_axis, **kwargs
    )


@configurable
def plot_spectrogram(
    spectrogram: typing.Union[torch.Tensor, np.ndarray],
    sample_rate: int = HParam(),
    frame_hop: int = HParam(),
    cmap: str = "turbo",
    y_axis: str = "linear",
    x_axis: str = "time",
    fmax: typing.Optional[float] = None,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Plot a spectrogram.

    Args:
        spectrogram (numpy.array or torch.FloatTensor [num_frames, num_frequencies])
        sample_rate: Sample rate for the signal.
        frame_hop: The frame hop in samples.
        **kwargs: Any additional keyword arguments are passed onto `librosa_display.specshow`.
    """
    assert len(spectrogram.shape) == 2, "Spectrogram must be 2-dimensional."
    figure = pyplot.figure()
    spectrogram = (
        spectrogram.detach().cpu().numpy() if torch.is_tensor(spectrogram) else spectrogram
    )
    spectrogram = np.transpose(spectrogram)
    fmax = fmax if fmax is None else min(fmax, float(sample_rate) / 2)
    librosa_display.specshow(
        spectrogram,
        hop_length=frame_hop,
        sr=sample_rate,
        cmap=cmap,
        y_axis=y_axis,
        x_axis=x_axis,
        fmax=fmax,
        **kwargs,
    )
    pyplot.colorbar(format="%.2f")
    pyplot.close(figure)
    return figure


# Comet HTML for ``log_html`` base styles.
_BASE_HTML_STYLING = """
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.css"
      type="text/css">
<style>
  body {
    background-color: #f4f4f5;
  }

  p {
    font-family: 'Roboto', system-ui, sans-serif;
    margin-bottom: .5em;
  }

  b {
    font-weight: bold
  }

  section {
    padding: 1.5em;
    border-bottom: 2px solid #E8E8E8;
    background: white;
  }
</style>
"""


def CometMLExperiment(
    project_name: typing.Optional[str] = None,
    experiment_key: typing.Optional[str] = None,
    workspace: typing.Optional[str] = None,
    **kwargs,
) -> comet_ml.Experiment:
    """Create a `comet_ml.Experiment` or `comet_ml.ExistingExperiment` object with several
    adjustments.

    Args:
        project_name
        experiment_key: Existing experiment identifier.
        workspace
        **kwargs: Other kwargs to pass to comet `Experiment` and / or `ExistingExperiment`
    """
    if lib.environment.has_untracked_files():
        raise ValueError(
            "Experiment is not reproducible, Comet does not track untracked files. "
            f"Please track these files via `git`:\n{lib.environment.get_untracked_files()}"
        )

    kwargs.update({"project_name": project_name, "workspace": workspace})
    if experiment_key is None:
        experiment = comet_ml.Experiment(**kwargs)
        experiment.log_html(_BASE_HTML_STYLING)
    else:
        experiment = comet_ml.ExistingExperiment(previous_experiment=experiment_key, **kwargs)

    experiment.log_other("last_git_commit_date", lib.environment.get_last_git_commit_date())
    experiment.log_other("git_branch", lib.environment.get_git_branch_name())
    experiment.log_other("has_git_patch", str(lib.environment.has_tracked_changes()))
    experiment.log_other("gpus", lib.environment.get_cuda_gpus())
    experiment.log_other("num_gpus", lib.environment.get_num_cuda_gpus())
    experiment.log_other("disks", lib.environment.get_disks())
    experiment.log_other("unique_cpus", lib.environment.get_unique_cpus())
    experiment.log_other("num_cpus", os.cpu_count())
    experiment.log_other("total_physical_memory", lib.environment.get_total_physical_memory())

    _add_comet_ml_set_step(experiment)
    _add_comet_ml_set_context(experiment)
    _add_comet_ml_log_epoch(experiment)
    _add_comet_ml_log_audio(experiment)
    _add_comet_ml_log_figures(experiment)
    _add_comet_ml_set_name(experiment)
    _add_comet_ml_add_tags(experiment)

    return experiment


def _add_comet_ml_set_step(experiment: comet_ml.Experiment):
    """Monkey patch `Experiment.set_step`.

    This adds the metric 'step/seconds_per_step', it measures the number of seconds each step takes.
    """
    _set_step = experiment.set_step
    last_step_time = None
    last_step = None

    def set_step(self, *args, **kwargs):
        return_ = _set_step(*args, **kwargs)

        nonlocal last_step_time
        nonlocal last_step

        if last_step_time is None and last_step is None:
            last_step_time = time.time()
            last_step = self.curr_step
        elif self.curr_step > last_step:
            seconds_per_step = (time.time() - last_step_time) / (self.curr_step - last_step)
            last_step_time = time.time()
            last_step = self.curr_step
            # NOTE: Ensure that the variable `last_step` is updated before `log_metric` is called.
            # This prevents infinite recursion via `self.curr_step > last_step`.
            self.log_metric("step/seconds_per_step", seconds_per_step)

        return return_

    # Learn more:
    # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
    experiment.set_step = types.MethodType(set_step, experiment)


def _add_comet_ml_set_context(experiment: comet_ml.Experiment):
    """Add `set_context` to `comet_ml.Experiment` to support a custom context."""

    @contextmanager
    def set_context(self, context):
        old_context = self.context
        self.context = context
        yield self
        self.context = old_context

    experiment.set_context = types.MethodType(set_context, experiment)


def _add_comet_ml_log_epoch(experiment: comet_ml.Experiment):
    """Monkey patch `Experiment.log_current_epoch` and `Experiment.log_epoch_end`.

    This adds the metrics 'steps_per_second' and 'epoch/steps_per_second'.
    """
    _log_current_epoch = experiment.log_current_epoch
    _log_epoch_end = experiment.log_epoch_end
    last_epoch_time = None
    last_epoch_step = None
    first_epoch_time = None
    first_epoch_step = None

    def log_current_epoch(self, *args, **kwargs):
        nonlocal last_epoch_time
        nonlocal last_epoch_step
        nonlocal first_epoch_time
        nonlocal first_epoch_step

        last_epoch_step = self.curr_step
        last_epoch_time = time.time()
        if first_epoch_time is None and first_epoch_step is None:
            first_epoch_step = self.curr_step
            first_epoch_time = time.time()

        return _log_current_epoch(*args, **kwargs)

    def log_epoch_end(self, *args, **kwargs):
        # NOTE: Logs an average `steps_per_second` for each epoch.
        if last_epoch_step is not None and last_epoch_time is not None:
            self.log_metric(
                "epoch/steps_per_second",
                (self.curr_step - last_epoch_step) / (time.time() - last_epoch_time),
            )

        # NOTE: Logs an average `steps_per_second` since the training started.
        if first_epoch_time is not None and first_epoch_step is not None:
            with experiment.set_context(None):
                self.log_metric(
                    "steps_per_second",
                    (self.curr_step - first_epoch_step) / (time.time() - first_epoch_time),
                )

        return _log_epoch_end(*args, **kwargs)

    experiment.log_current_epoch = types.MethodType(log_current_epoch, experiment)
    experiment.log_epoch_end = types.MethodType(log_epoch_end, experiment)


def _add_comet_ml_log_audio(experiment: comet_ml.Experiment):
    """Add `log_audio` to `comet_ml.Experiment` to support logging audio.

    NOTE: This assumes that the HTML has `_BASE_HTML_STYLING`.
    """

    def _upload_audio(file_name: str, data: typing.Union[np.ndarray, torch.Tensor]) -> str:
        """Upload the audio and return the URL."""
        file_ = io.BytesIO()
        lib.audio.write_audio(file_, data)
        asset = experiment.log_asset(file_, file_name=file_name)
        return asset["web"] if asset is not None else asset

    def log_audio(
        self,
        audio: typing.Dict[str, typing.Union[np.ndarray, torch.Tensor]] = {},
        **kwargs,
    ):
        """Audio with related metadata to Comet in the HTML tab.

        Args:
            audio
            **kwargs: Additional metadata to include.
        """
        items = [f"<p><b>Step:</b> {self.curr_step}</p>"]
        param_to_label = lambda s: s.title().replace("_", " ")
        items.extend([f"<p><b>{param_to_label(k)}:</b> {v}</p>" for k, v in kwargs.items()])
        for key, data in audio.items():
            name = param_to_label(key)
            file_name = f"step={self.curr_step},name={name},experiment={self.get_key()}"
            url = _upload_audio(file_name, data)
            items.append(f"<p><b>{name}:</b></p>")
            items.append(f'<audio controls preload="metadata" src="{url}"></audio>')
        experiment.log_html("<section>{}</section>".format("\n".join(items)))

    experiment.log_audio = types.MethodType(log_audio, experiment)


def _add_comet_ml_log_figures(experiment: comet_ml.Experiment):
    """ Add `log_figures` to `comet_ml.Experiment` to support logging multiple figures. """

    def log_figures(self, dict_: typing.Dict[str, matplotlib.figure.Figure], **kwargs):
        """ Log multiple figures from `dict_` via `experiment.log_figure`. """
        return [self.log_figure(k, v, **kwargs) for k, v in dict_.items()]

    experiment.log_figures = types.MethodType(log_figures, experiment)


def _add_comet_ml_set_name(experiment: comet_ml.Experiment):
    """ Monkey patch `Experiment.set_name` with additional logging. """
    _set_name = experiment.set_name

    def set_name(self, name, *args, **kwargs):
        logger.info('Experiment name set to "%s"', name)
        return _set_name(name, *args, **kwargs)

    experiment.set_name = types.MethodType(set_name, experiment)


def _add_comet_ml_add_tags(experiment: comet_ml.Experiment):
    """ Monkey patch `Experiment.add_tags` with additional logging. """
    _add_tags = experiment.add_tags

    def add_tags(self, tags, *args, **kwargs):
        logger.info("Added tags to experiment: %s", tags)
        return _add_tags(tags, *args, **kwargs)

    experiment.add_tags = types.MethodType(add_tags, experiment)
