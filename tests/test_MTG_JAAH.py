# -*- coding: utf-8 -*-

import numpy as np

from mirdata import MTG_JAAH, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '0'
    data_home = '../tests/resources/mir_datasets/MTG_JAAH'
    track = MTG_JAAH.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': '../tests/resources/mir_datasets/MTG_JAAH/audio/airegin.flac',
        'ann_path': '../tests/resources/mir_datasets/MTG_JAAH/annotations/airegin.json',
        'labs_path': '../tests/resources/mir_datasets/MTG_JAAH/labs/airegin.lab',
        'track_id': '0'
    }

    expected_property_types = {
        'chords': utils.ChordData,
        'beats': utils.BeatData,
        'sections': utils.SectionData,
        'key': str,
        'artist': str,
        'tuning': float,
        'mbid': str,
        'metre': str,
        'duration': float
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, 'sample rate {} is not 44100'.format(sr)


