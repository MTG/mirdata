# -*- coding: utf-8 -*-

import numpy as np
from mirdata import salami, utils
from tests.test_utils import run_track_tests


def test_track():

    default_trackid = 'carnatic_1'
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = salami.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'carnatic_1',
        'iam_style': "carnatic",
        'audio_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mp3',
        'ctonic_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                       'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.ctonic.txt',
        'pitch_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch.txt',
        'pitch_vocal_path': None,
        'bpm_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                    'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.bpm-manual.txt',
        'tempo_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.tempo-manual.txt',
        'sama_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                     'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.sama-manual.txt',
        'sections_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                         'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.sections-manual-p.txt',
        'phrases_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                        'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mphrases-manual.txt',
        'metadata_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                         'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.json',
        'raaga': [{'uuid': '42dd0ccb-f92a-4622-ae5d-a3be571b4939', 'name': 'Śrīranjani'}],
        'form': [{'name': 'Kriti'}],
        'title': "Bhuvini Dasudane",
        'work': [{'mbid': '4d05ce9b-c45e-4c85-9eca-941d68b61132', 'title': 'Bhuvini Dasudane'}],
        'taala': [{'uuid': 'c788c38a-b53a-48cb-b7bf-d11769260c4d', 'name': 'Ādi'}],
        'album_artists': [{'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}],
        'mbid': "9f5a5452-14cb-4af0-9289-4833854ee60d",
        'artists': [{'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'attributes': '',
                     'lead': False, 'artist': {'mbid': '19f93366-5d58-47f1-bc4f-9225ac7af6ba', 'name': 'N Guruprasad'}},
                    {'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'},
                     'attributes': '', 'lead': False,
                     'artist': {'mbid': '39c1d741-6154-418b-bf4b-12c77ba13873', 'name': 'Srimushnam V Raja Rao'}},
                    {'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'attributes': '',
                     'lead': False,
                     'artist': {'mbid': 'a2df55e3-d141-4767-862e-77adca691d4b', 'name': 'B.U. Ganesh Prasad'}},
                    {'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'},
                     'attributes': 'lead vocals', 'lead': True,
                     'artist': {'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}
                     }],
        'concert': [{'mbid': '0816586d-c83e-4c79-a0aa-9b0e578f408d', 'title': 'Cherthala Ranganatha Sharma at Arkay'}],
    }

    expected_property_types = {
        'audio': (np.ndarray, float),
        'bpm': utils.TempoData,
        # 'tempo': TempoData
        'phrases': utils.SectionData,
        'pitch': utils.F0Data,
        'pitch_vocal': utils.F0Data,
        'sama': utils.SectionData,
        'sections': utils.SectionData,
        'tonic': int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (89856,)


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/Salami'
    track = salami.Track('2', data_home=data_home)
    jam = track.to_jams()

    segments = jam.search(namespace='multi_segment')[0]['data']
    assert [segment.time for segment in segments] == [
        0.0,
        0.0,
        0.464399092,
        0.464399092,
        5.191269841,
        14.379863945,
        254.821632653,
        258.900453514,
        263.205419501,
        263.205419501,
    ]
    assert [segment.duration for segment in segments] == [
        0.464399092,
        0.464399092,
        13.915464853,
        4.726870749000001,
        249.630362812,
        248.82555555599998,
        4.078820860999997,
        4.304965987000003,
        1.6797959180000248,
        1.6797959180000248,
    ]
    assert [segment.value for segment in segments] == [
        {'label': 'Silence', 'level': 0},
        {'label': 'Silence', 'level': 1},
        {'label': 'A', 'level': 0},
        {'label': 'b', 'level': 1},
        {'label': 'b', 'level': 1},
        {'label': 'B', 'level': 0},
        {'label': 'ab', 'level': 1},
        {'label': 'ab', 'level': 1},
        {'label': 'Silence', 'level': 0},
        {'label': 'Silence', 'level': 1},
    ]
    assert [segment.confidence for segment in segments] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    assert jam['file_metadata']['title'] == 'For_God_And_Country'
    assert jam['file_metadata']['artist'] == 'The_Smashing_Pumpkins'


def test_load_sections():
    # load a file which exists
    sections_path = (
        'tests/resources/mir_datasets/Salami/'
        + 'salami-data-public-hierarchy-corrections/annotations/2/parsed/textfile1_uppercase.txt'
    )
    section_data = salami.load_sections(sections_path)

    # check types
    assert type(section_data) == utils.SectionData
    assert type(section_data.intervals) is np.ndarray
    assert type(section_data.labels) is list

    # check valuess
    assert np.array_equal(
        section_data.intervals[:, 0],
        np.array([0.0, 0.464399092, 14.379863945, 263.205419501]),
    )
    assert np.array_equal(
        section_data.intervals[:, 1],
        np.array([0.464399092, 14.379863945, 263.205419501, 264.885215419]),
    )
    assert np.array_equal(
        section_data.labels, np.array(['Silence', 'A', 'B', 'Silence'])
    )

    # load none
    section_data_none = salami.load_sections(None)
    assert section_data_none is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/Salami'
    metadata = salami._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['2'] == {
        'source': 'Codaich',
        'annotator_1_id': '5',
        'annotator_2_id': '8',
        'duration': 264,
        'title': 'For_God_And_Country',
        'artist': 'The_Smashing_Pumpkins',
        'annotator_1_time': '37',
        'annotator_2_time': '45',
        'class': 'popular',
        'genre': 'Alternative_Pop___Rock',
    }

    none_metadata = salami._load_metadata('asdf/asdf')
    assert none_metadata is None
