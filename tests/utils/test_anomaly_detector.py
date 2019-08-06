from src.utils.anomaly_detector import AnomalyDetector


def test_anomaly_detector():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.step(2)


def test_anomaly_detector__type_high():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_HIGH)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.step(2)


def test_anomaly_detector__type_low():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_LOW)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert not anomaly_detector.step(2)


def test_anomaly_detector__type_both():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_BOTH)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.max_deviation < 1
    assert anomaly_detector.step(2)
    assert anomaly_detector.step(0)
    assert anomaly_detector.step(float('nan'))
    assert anomaly_detector.step(float('inf'))
