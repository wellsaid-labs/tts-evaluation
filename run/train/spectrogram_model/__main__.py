import logging
import pathlib
import typing
from functools import partial
from unittest.mock import MagicMock

import config as cf
import torch
import torch.optim

import lib
from run._config import (
    DEV_SPEAKERS,
    NUM_FRAME_CHANNELS,
    RANDOM_SEED,
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    get_config_label,
)
from run._utils import Dataset
from run.data._loader.structures import Language
from run.train._utils import (
    CometMLExperiment,
    resume_experiment,
    run_workers,
    set_run_seed,
    start_experiment,
)
from run.train.spectrogram_model import _data, _metrics, _worker

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    import typer

    app = typer.Typer()
else:
    try:
        import typer

        app = typer.Typer()
    except (ModuleNotFoundError, NameError):
        app = MagicMock()
        typer = MagicMock()
        logger.info("Ignoring optional `typer` dependency.")


ENGLISH_TEST_CASES = [
    # NOTE: These statements have a mix of heteronyms, initialisms, hard words (locations,
    # medical terms, technical terms), etc for testing pronunciation.
    "For more updates on covid nineteen, please contact us via the URL at the bottom of the "
    "screen, or visit our office in Seattle at the address shown here.",
    "I've listed INTJ on my resume because it's important for me that you understand how I "
    "conduct myself in stressful situations.",
    "The website is live and you can access your records via the various APIs slash URLs or use "
    "the Studio as an alternate avenue.",
    "The nurses will resume the triage conduct around the oropharyngeal and test for "
    "tachydysrhythmia to ensure the patient lives another day.",
    "Access your clusters using the Kubernetes API. You can alternate between the CLI and the "
    "web interface.",
    "Live from Seattle, it's AIQTV, with the governor's special address on the coronavirus. Don't "
    "forget to record this broadcast for viewing later.",
    "Let's add a row on our assay tracking sheet so we can build out the proper egress "
    "measurements.",
    "Hello! Can you put this contractor into a supervisory role?",
    # NOTE: These test various initialisms
    "Each line will have GA Type as Payment, Paid Amount along with PAC, and GA Code.",
    "Properly use and maintain air-line breathing systems and establish a uniform procedure "
    "for all employees, for both LACC and LCLA contractors, to follow when working jobs that "
    "require the use of fresh air.",
    "QCBS is a method of selecting transaction advisors based on both the quality of their "
    "technical proposals and the costs shown in their financial proposals.",
    "HSPs account for fifteen to twenty percent of the population.",
    "We used to have difficulty with AOA and AMA, but now we are A-okay.",
    "As far as AIs go, ours is pretty great! (",
    # NOTE: These questions each have a different expected inflection.
    "If you can instantly become an expert in something, what would it be?",
    "What led to the two of you having a disagreement?",
    "Why do some words sound funny to us?",
    "What are your plans for dealing with it?",
    "There may be times where you have to RDP to a node and manually collect logs for some "
    "reason. So, another question you may have is, exactly where on disk are all these logs?",
    "How common are HSPs?",
    "If you could rid the world of one thing, what would it be?",
    "What childish things do you still do as an adult?",
    "If you were to perform in the circus, what would you do?",
    # NOTE: All these questions should have an upward inflection at the end.
    "Have you ever hidden a snack so that nobody else would find it and eat it first?",
    "Can fish see air like we see water?",
    "Are you a messy person?",
    "Did you have cats growing up?",
    "Do you consider yourself an adventurous person?",
    "Do you have any weird food combos?",
    "Do you respond to texts fast?",
    "Have you ever been stalked by an animal that later became your pet?",
    "If you have made it this far, do you relate to any of these signs? Are you a highly "
    "sensitive person?",
    "Have you started, but not found success, with a platform requiring monthly payments?",
    "When deciding between organic and non-organic coffees, is the price premium worth it?",
    "Can you make yourself disappear?",
    "Do mice really eat cheese?",
    "Do you believe in any conspiracy theories?",
    "Have elves always lived at the North Pole?",
    "Have you ever been on the radio?",
    "Have you ever done something embarrassing in front of the office CCTV cameras?",
    "In your opinion, are giant spiders worse than giant chickens?",
    "What is the process for making your favorite dish?",
    "Would you like to be part of the UK Royal Family?",
    "Did you ever try DIY projects?",
    "Can people from NASA catch the flu?",
    "Do you watch ESPN at night?",
    "Will AI replace humans?",
    "Can our AI say AI?",
    # NOTE: Test cases with a variety of lengths, respellings, and punctuation marks.
    "WellSaid Labs.",
    "Livingroom",
    "Ophthalmologist",
    "ACLA",
    "ACLA.",  # NOTE: `ACLA` sometimes gets cut-off, this is a test to see how a period affects it.
    "NASA",
    "Why?",
    'Ready to find out ""more""?',
    "Thisss isrealy awhsome.",
    "Topic two:     Is an NRA right for my rate?.",
    'Internet Assigned Numbers Authority ("""I-eigh n Eigh""")',
    '"""G-E-ran""" is an abbreviation for GSM EDGE',
    "epidermolysis bullosa (ep-ih-dur-MOL-uh-sis buhl-LOE-sah) (epi-dermo-lysiss) is a group of",
    "Harry lay in his dark cupboard much later, wishing he had a watch. He didn't know what time "
    "it was and he couldn't be sure the Dursleys were asleep yet. Until they were, he couldn't "
    "risk sneaking to the kitchen for some food. He'd lived with the Dursleys almost ten years, "
    "ten miserable years, as long as he could remember, ever since he'd been a baby and his "
    "parents had died in that car crash. He couldn't remember being in the car when his parents "
    "had died. Sometimes, when he strained his memory during long hours in his cupboard, he came "
    "up with a strange vision: a blinding flash of green light and a burning pain on his "
    "forehead. This, he supposed, was the crash, though he couldn't imagine where all the green "
    "light came from. He couldn't remember his parents at all. His aunt and uncle never spoke "
    "about them, and of course he was forbidden to ask questions. There were no photographs of "
    "them in the house. When he had been younger, Harry had dreamed and dreamed of some unknown "
    "relation coming to take him away, but it had never happened; the Dursleys were his only "
    "family. Yet sometimes he thought (or maybe hoped) that strangers in the street seemed to "
    "know him. Very strange strangers they were, too.",
    # NOTE: Test respellings
    "I see in “Happening at |\\se\\FOHR\\u\\|” I have two new brands requesting store-led events "
    "for the same day.",
    "Welcome to the |\\su\\LAHR\\es\\| Injury and Illness Prevention Program Training.",
    "The |\\pur\\AY\\toh\\| principle was named after Italian economist Vilfredo "
    "|\\pu\\RAY\\toh\\|.",
    "We would like to nominate |\\AY\\vu\\| for her phenomenal recordings.",
    "To use your self-help AI, please enable the Affirmations feature on the |\\KAHN\\səl\\| so "
    "that you can |\\kun\\SOHL\\| yourself.",
    "Too much sand? Tired of cacti? |\\dee\\ZURT\\| the |\\DEZ\\urt\\| now, with caravan "
    "adventures!",
    "If you want to get the good food at the |\\bu\\FAY\\|, you have to be willing to "
    "|\\BUF\\et\\| and punch your way to the front of the line.",
    "Does |\\BEE\\ə\\loh\\ZHEEK\\| |\\ru\\SHƏRSH\\| really work?",
]
TEST_CASES = [(Language.ENGLISH, t) for t in ENGLISH_TEST_CASES]


def _make_configuration(train_dataset: Dataset, dev_dataset: Dataset, debug: bool) -> cf.Config:
    """Make additional configuration for spectrogram model training."""
    train_size = sum(sum(p.segmented_audio_length() for p in d) for d in train_dataset.values())
    dev_size = sum(sum(p.segmented_audio_length() for p in d) for d in dev_dataset.values())
    ratio = train_size / dev_size
    logger.info("The training dataset is approx %fx bigger than the development dataset.", ratio)
    train_batch_size = 28 if debug else 56
    batch_size_ratio = 4
    dev_batch_size = train_batch_size * batch_size_ratio
    dev_steps_per_epoch = 1 if debug else 64
    train_steps_per_epoch = int(round(dev_steps_per_epoch * batch_size_ratio * ratio))
    train_steps_per_epoch = 1 if debug else train_steps_per_epoch
    assert train_batch_size % lib.distributed.get_device_count() == 0
    assert dev_batch_size % lib.distributed.get_device_count() == 0

    return {
        set_run_seed: cf.Args(seed=RANDOM_SEED),
        # NOTE: We expect users to respell approx 5 - 10% of words.
        _data.make_batch: cf.Args(respell_prob=0.1),
        _worker._State._get_optimizers: cf.Args(
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
            # SOURCE (Tacotron 2):
            # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
            optimizer=torch.optim.AdamW,
            exclude_from_decay=_worker.exclude_from_decay,
        ),
        _worker._run_step: cf.Args(
            # NOTE: This scalar calibrates the loss so that it's scale is similar to Tacotron-2.
            spectrogram_loss_scalar=1 / 100,
            # NOTE: This value is the average spectrogram length in the training dataset.
            average_spectrogram_length=58.699,
            # NOTE: This starts to decay the stop token loss as soon as it converges so it doesn't
            # overfit. Also, this ensures that the model doesn't unnecessarily prioritize the stop
            # token loss when it has already converged.
            stop_token_loss_multiplier=partial(
                lib.optimizers.exponential_decay_lr_multiplier_schedule,
                warmup=0,
                start_decay=10_000,
                end_decay=50_000,
                multiplier=0.001,
            ),
        ),
        _worker._get_data_loaders: cf.Args(
            # SOURCE: Tacotron 2
            # To train the feature prediction network, we apply the standard maximum-likelihood
            # training procedure (feeding in the correct output instead of the predicted output on
            # the decoder side, also referred to as teacher-forcing) with a batch size of 64 on a
            # single GPU.
            # NOTE: Batch size parameters set after experimentation on a 2 Px100 GPU.
            train_batch_size=train_batch_size,
            dev_batch_size=dev_batch_size,
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=int(dev_steps_per_epoch),
            train_get_weight=_data.train_get_weight,
            dev_get_weight=_data.dev_get_weight,
            num_workers=2,
            prefetch_factor=2 if debug else 10,
        ),
        _worker._visualize_select_cases: cf.Args(
            cases=TEST_CASES, speakers=DEV_SPEAKERS, num_cases=15
        ),
        _metrics.Metrics._get_model_metrics: cf.Args(num_frame_channels=NUM_FRAME_CHANNELS),
        # NOTE: Based on the alignment visualizations, if the maximum alignment is less than 30%
        # then a misalignment has likely occured.
        _metrics.get_num_small_max: cf.Args(threshold=0.3),
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer with β1 = 0.9, β2 = 0.999, eps = 10−6 learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        # NOTE: No L2 regularization performed better based on Comet experiments in March 2020.
        torch.optim.AdamW: cf.Args(
            eps=10 ** -6,
            weight_decay=0.01,
            lr=10 ** -3,
            amsgrad=False,
            betas=(0.9, 0.999),
        ),
    }


def _run_app(
    checkpoints_directory: pathlib.Path,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path],
    cli_config: cf.Config,
    debug: bool,
):
    """Run spectrogram model training.

    TODO: PyTorch-Lightning makes strong recommendations to not use `spawn`. Learn more:
    https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#distributed-data-parallel
    https://github.com/PyTorchLightning/pytorch-lightning/pull/2029
    https://github.com/PyTorchLightning/pytorch-lightning/issues/5772
    Also, it's less normal to use `spawn` because it wouldn't work with multiple nodes, so
    we should consider using `torch.distributed.launch`.
    TODO: Should we consider setting OMP num threads similarly:
    https://github.com/pytorch/pytorch/issues/22260
    """
    cf.add(_make_configuration(train_dataset, dev_dataset, debug))
    cf.add(cli_config)
    comet.log_parameters({get_config_label(k): v for k, v in cf.log(lambda x: x).items()})
    return run_workers(
        _worker.run_worker, comet, checkpoint, checkpoints_directory, train_dataset, dev_dataset
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def resume(
    context: typer.Context,
    checkpoint: typing.Optional[pathlib.Path] = typer.Argument(
        None, help="Checkpoint file to restart training from.", exists=True, dir_okay=False
    ),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """Resume training from CHECKPOINT. If CHECKPOINT is not given, the most recent checkpoint
    file is loaded."""
    args = resume_experiment(SPECTROGRAM_MODEL_EXPERIMENTS_PATH, checkpoint, debug=debug)
    cli_config = cf.parse_cli_args(context.args)
    _run_app(*args, cli_config, debug)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    context: typer.Context,
    project: str = typer.Argument(..., help="Experiment project name."),
    name: str = typer.Argument("", help="Experiment name."),
    tags: typing.List[str] = typer.Option([], help="Experiment tags."),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """Start a training run in PROJECT named NAME with TAGS."""
    args = start_experiment(SPECTROGRAM_MODEL_EXPERIMENTS_PATH, project, name, tags, debug=debug)
    cli_config = cf.parse_cli_args(context.args)
    _run_app(*args, None, cli_config, debug)


if __name__ == "__main__":  # pragma: no cover
    app()
