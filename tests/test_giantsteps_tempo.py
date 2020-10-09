# -*- coding: utf-8 -*-

import numpy as np

from mirdata import giantsteps_tempo, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '113'
    data_home = 'tests/resources/mir_datasets/GiantSteps_tempo'
    track = giantsteps_tempo.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': 'tests/resources/mir_datasets/GiantSteps_tempo/audio/28952.LOFI.mp3',
        'annotation_v1_path': 'tests/resources/mir_datasets/GiantSteps_tempo/giantsteps-tempo-dataset'
                              '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams',
        'annotation_v2_path': 'tests/resources/mir_datasets/GiantSteps_tempo/giantsteps-tempo-dataset'
                              '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations_v2/jams/28952.LOFI.jams',
        'title': '28952',
        'track_id': '113',
    }

    expected_property_types = {
        'tempo': list,
        'tempo_v2': list,
        'genre': str
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 22050, 'sample rate {} is not 22050'.format(sr)
    print(audio.shape)
    assert audio.shape == (2646720,), 'audio shape {} was not (5294592,)'.format(
        audio.shape
    )


def test_load_genre():
    genre_path = 'tests/resources/mir_datasets/GiantSteps_tempo/giantsteps-tempo-dataset' \
                 '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams'

    genre_data = giantsteps_tempo.load_genre(genre_path)

    assert type(genre_data) == str

    assert genre_data == "trance"

    assert giantsteps_tempo.load_tempo(None) is None


def test_load_tempo():
    tempo_path = (
        'tests/resources/mir_datasets/GiantSteps_tempo/giantsteps-tempo-dataset'
        '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams'
    )
    tempo_data = giantsteps_tempo.load_tempo(tempo_path)

    assert type(tempo_data) == list

    assert tempo_data == [utils.TempoData(time=120.0, duration=1.0, value=137.6, confidence=0.0)]

    tempo_path = (
        'tests/resources/mir_datasets/GiantSteps_tempo/giantsteps-tempo-dataset'
        '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations_v2/jams/28952.LOFI.jams'
    )
    tempo_data = giantsteps_tempo.load_tempo(tempo_path)

    assert type(tempo_data) == list

    assert tempo_data == [utils.TempoData(time=120.0, duration=0.30927835051546393, value=77.0, confidence=0.0),
                          utils.TempoData(time=120.0, duration=0.6907216494845361, value=139.0, confidence=0.0)]

    assert giantsteps_tempo.load_tempo(None) is None
