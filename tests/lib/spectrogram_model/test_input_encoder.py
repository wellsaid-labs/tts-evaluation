import torch

import lib.spectrogram_model.input_encoder


def test_input_encoder():
    """ Test `lib.spectrogram_model.input_encoder.InputEncoder` handles a basic case. """
    graphemes = ["abc", "def"]
    phonemes = ["ˈ|eɪ|b|ˌ|iː|s|ˈ|iː|", "d|ˈ|ɛ|f"]
    phoneme_separator = "|"
    speakers = [lib.datasets.MARK_ATHERLAY, lib.datasets.MARY_ANN]
    encoder = lib.spectrogram_model.input_encoder.InputEncoder(
        graphemes, phonemes, speakers, phoneme_separator
    )
    input_ = ("a", "ˈ|eɪ", lib.datasets.MARK_ATHERLAY)
    assert encoder._get_case("A") == encoder._CASE_LABELS[0]
    assert encoder._get_case("a") == encoder._CASE_LABELS[1]
    assert encoder._get_case("1") == encoder._CASE_LABELS[2]
    encoded = encoder.encode(input_)
    assert torch.equal(encoded[0], torch.tensor([5]))
    assert torch.equal(encoded[1], torch.tensor([1]))
    assert torch.equal(encoded[2], torch.tensor([5, 6]))
    assert torch.equal(encoded[3], torch.tensor([0]))
    assert encoder.decode(encoded) == input_
