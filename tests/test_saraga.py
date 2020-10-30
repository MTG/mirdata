# -*- coding: utf-8 -*-

import numpy as np
from mirdata import saraga, utils
from tests.test_utils import run_track_tests

TEST_DATA_HOME = '/Users/genisplaja/Desktop/genis-datasets/'


def test_track():

    default_trackid = 'carnatic_1'
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'carnatic_1',
        'iam_style': "carnatic",
        'audio_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mp3',
        'ctonic_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                       'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.ctonic.txt',
        'pitch_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch.txt',
        'pitch_vocal_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch-vocal.txt',
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
        'phrases': utils.EventData,
        'pitch': utils.F0Data,
        'pitch_vocal': utils.F0Data,
        'sama': utils.SectionData,
        'sections': utils.SectionData,
        'tonic': float,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=TEST_DATA_HOME)
    metadata = saraga._load_metadata(track.metadata_path)
    jam = track.to_jams()

    # Tonic
    assert jam['sandbox'].tonic == 201.740890

    # Pitch
    pitch = jam.search(namespace='pitch_contour')[0]['data']
    assert len(pitch) == 69603
    assert pitch[0].time == 0
    assert pitch[0].duration == 0.0
    assert pitch[0].value == {
        'index': 0, 'frequency': 0.0, 'voiced': False
    }
    assert pitch[0].confidence == 0.0

    assert pitch[3000].time == 13.3333333
    assert pitch[3000].duration == 0.0
    assert pitch[3000].value == {
        'index': 0, 'frequency': 223.8473663, 'voiced': True
    }
    assert pitch[3000].confidence == 1.0

    pitch_vocal = jam.search(namespace='pitch_contour')[1]['data']
    assert len(pitch_vocal) == 69603
    assert pitch_vocal[0].time == 0
    assert pitch_vocal[0].duration == 0.0
    assert pitch_vocal[0].value == {
        'index': 0, 'frequency': 0.0, 'voiced': False
    }
    assert pitch_vocal[0].confidence == 0.0

    assert pitch_vocal[3000].time == 13.3333333
    assert pitch_vocal[3000].duration == 0.0
    assert pitch_vocal[3000].value == {
        'index': 0, 'frequency': 223.8473663, 'voiced': True
    }
    assert pitch_vocal[3000].confidence == 1.0

    # Tempo TODO

    # Sama
    sama = jam.search(namespace='segment_open')[0]['data']
    # assert

    # Sections
    sections = jam.search(namespace='segment_open')[1]['data']

    # Phrases
    phrases = jam.search(namespace='tag_open')[0]['data']

    # Metadata
    metadata = jam['sandbox'].metadata


def main():
    data_home = TEST_DATA_HOME
    # data_home = '../tests/resources/mir_datasets/Saraga/saraga1.0'
    ids = saraga.track_ids()
    data = saraga.load(data_home)

    track_carnatic = data['carnatic_1']
    # print(track_carnatic.sama)
    test_to_jams()


if __name__ == '__main__':
    main()