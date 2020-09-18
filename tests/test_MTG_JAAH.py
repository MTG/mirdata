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


def test_load_beats():
    beats_path = (
        'tests/resources/mir_datasets/MTG_JAAH'
        + 'tests/resources/mir_datasets/MTG_JAAH/annotations/airegin.json'
    )
    beat_data = MTG_JAAH.load_beats(beats_path)

    assert type(beat_data) == utils.BeatData, 'beat_data is not type utils.BeatData'
    assert (
        type(beat_data.beat_times) == np.ndarray
    ), 'beat_data.beat_times is not an np.ndarray'
    assert (
        type(beat_data.beat_positions) == np.ndarray
    ), 'beat_data.beat_positions is not an np.ndarray'

    assert np.array_equal(
        beat_data.beat_times,
        np.array([13.249, 13.959, 14.416, 14.965, 15.453, 15.929, 16.428]),
    ), 'beat_data.beat_times different than expected'
    assert np.array_equal(
        beat_data.beat_positions, np.array([2, 3, 4, 1, 2, 3, 4])
    ), 'beat_data.beat_positions different from expected'

    assert MTG_JAAH.load_beats(None) is None, 'load_beats(None) should return None'


def test_load_chords():
    chords_path = (
        'tests/resources/mir_datasets/Beatles/annotations/chordlab/'
        + 'The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab'
    )
    chord_data = MTG_JAAH.load_chords(chords_path)

    assert type(chord_data) == utils.ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list

    assert np.array_equal(
        chord_data.intervals[:, 0], np.array([0.000000, 4.586464, 6.989730])
    )
    assert np.array_equal(
        chord_data.intervals[:, 1], np.array([0.497838, 6.989730, 9.985104])
    )
    assert np.array_equal(chord_data.labels, np.array(['N', 'E:min', 'G']))

    assert MTG_JAAH.load_chords(None) is None


def test_load_sections():
    sections_path = (
        'tests/resources/mir_datasets/Beatles/annotations/seglab/'
        + 'The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab'
    )
    section_data = MTG_JAAH.load_sections(sections_path)

    assert type(section_data) == utils.SectionData
    assert type(section_data.intervals) == np.ndarray
    assert type(section_data.labels) == list

    assert np.array_equal(section_data.intervals[:, 0], np.array([0.000000, 0.465]))
    assert np.array_equal(section_data.intervals[:, 1], np.array([0.465, 14.931]))
    assert np.array_equal(section_data.labels, np.array(['silence', 'intro']))

    assert MTG_JAAH.load_sections(None) is None


