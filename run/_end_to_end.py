import typing

import numpy
import torch

import lib
import run


def text_to_speech(
    input_encoder: run.train.spectrogram_model._worker.InputEncoder,
    spec_model: lib.spectrogram_model.SpectrogramModel,
    sig_model: lib.signal_model.SignalModel,
    script: str,
    speaker: run.data._loader.Speaker,
    session: run.data._loader.Session,
    split_size: int = 32,
) -> numpy.ndarray:
    """Run TTS end-to-end.

    TODO: Add an end-to-end function for stream TTS.
    TODO: Add an end-to-end function for batch TTS.
    """
    script = lib.text.normalize_vo_script(script)
    nlp = lib.text.load_en_core_web_md(disable=("parser", "ner"))
    doc = nlp(script)
    phonemes = typing.cast(str, lib.text.grapheme_to_phoneme(doc))
    decoded = run.train.spectrogram_model._data.DecodedInput(
        script, phonemes, speaker, (speaker, session)
    )
    encoded = input_encoder.encode(decoded)
    params = lib.spectrogram_model.Params(
        tokens=encoded.phonemes, speaker=encoded.speaker, session=encoded.session
    )
    preds = spec_model(params=params, mode=lib.spectrogram_model.Mode.INFER)
    preds = typing.cast(lib.spectrogram_model.Infer, preds)
    preds = typing.cast(lib.spectrogram_model.Infer, preds)
    splits = preds.frames.split(split_size)
    predicted = list(
        lib.signal_model.generate_waveform(sig_model, splits, encoded.speaker, encoded.session)
    )
    predicted = typing.cast(torch.Tensor, torch.cat(predicted, dim=-1))
    return predicted.detach().numpy()
