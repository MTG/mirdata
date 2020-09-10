# -*- coding: utf-8 -*-

import numpy as np

from mirdata import beatles, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '0'
    data_home = 'tests/resources/mir_datasets/MTG_JAAH'
    track = beatles.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': 'tests/resources/mir_datasets/Beatles/'
        + 'audio/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.wav',
        'beats_path': 'tests/resources/mir_datasets/Beatles/'
        + 'annotations/beat/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.txt',
        'chords_path': 'tests/resources/mir_datasets/Beatles/'
        + 'annotations/chordlab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab',
        'keys_path': 'tests/resources/mir_datasets/Beatles/'
        + 'annotations/keylab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab',
        'sections_path': 'tests/resources/mir_datasets/Beatles/'
        + 'annotations/seglab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab',
        'title': '11_-_Do_You_Want_To_Know_A_Secret',
        'track_id': '0111',
    }

    expected_property_types = {
        'beats': utils.BeatData,
        'chords': utils.ChordData,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, 'sample rate {} is not 44100'.format(sr)
    assert audio.shape == (44100 * 2,), 'audio shape {} was not (88200,)'.format(
        audio.shape
    )

    track = beatles.Track('10212')
    assert track.beats is None, 'expected track.beats to be None, got {}'.format(
        track.beats
    )
    assert track.key is None, 'expected track.key to be None, got {}'.format(track.key)



def test_fix_newpoint():
    beat_positions1 = np.array(['4', '1', '2', 'New Point', '4'])
    new_beat_positions1 = beatles._fix_newpoint(beat_positions1)
    assert np.array_equal(new_beat_positions1, np.array(['4', '1', '2', '3', '4']))

    beat_positions2 = np.array(['1', '2', 'New Point'])
    new_beat_positions2 = beatles._fix_newpoint(beat_positions2)
    assert np.array_equal(new_beat_positions2, np.array(['1', '2', '3']))

    beat_positions3 = np.array(['New Point', '2', '3'])
    new_beat_positions3 = beatles._fix_newpoint(beat_positions3)
    assert np.array_equal(new_beat_positions3, np.array(['1', '2', '3']))
