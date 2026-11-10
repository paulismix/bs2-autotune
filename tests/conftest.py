"""Shared fixtures.  Slow end-to-end tests run only with ``pytest --runslow``."""

import pytest

from bs2autotune import engine, features
from bs2autotune.engine import Note, Phrase


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="also run the slow end-to-end tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end test (use --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def compiled():
    engine.warmup()
    features.warmup()


@pytest.fixture(scope="session")
def phrase():
    notes = [Note(0.05, 0.45, 28), Note(0.5, 0.9, 28), Note(1.0, 1.4, 31), Note(1.5, 2.4, 26),
             Note(2.5, 2.7, 28), Note(2.75, 3.3, 33)]
    return Phrase(notes, 3.8)


def bass_line():
    """A 11 s bass line: eighth notes at 110 bpm with rests, repeats and octave jumps."""
    beat = 60 / 110 / 2
    pitches = [28, 28, 40, 28, 31, 31, 33, 28, 28, 26, 26, 38, 33, 31, 28, 28, 35, 35, 33, 31, 28, 28, 28, 40, 31,
               33, 35, 36, 28, 28, 26, 24]
    notes = []
    t = 0.3
    for i, p in enumerate(pitches):
        dur = beat * (0.9 if i % 4 else 0.6) if i % 7 else beat * 1.9
        notes.append(Note(t, t + dur * 0.92, p))
        t += beat if i % 7 else beat * 2
    return Phrase(notes, t + 0.5)


@pytest.fixture(scope="session")
def line():
    return bass_line()
