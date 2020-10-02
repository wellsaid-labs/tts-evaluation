from src.bin.evaluate.audio_preprocessing import main


def test_main():
    audio_path = 'tests/_test_data/test_audio/rate(lj_speech,24000).wav'
    main(audio_path)
