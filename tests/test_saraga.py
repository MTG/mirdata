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
    assert len(pitch_vocal) == 187806
    assert pitch_vocal[0].time == 0
    assert pitch_vocal[0].duration == 0.0
    assert pitch_vocal[0].value == {
        'index': 0, 'frequency': 0.0, 'voiced': False
    }
    assert pitch_vocal[0].confidence == 0.0

    assert pitch_vocal[8000].time == 23.219954648526077
    assert pitch_vocal[8000].duration == 0.0
    assert pitch_vocal[8000].value == {
        'index': 0, 'frequency': 175.6268310546875, 'voiced': True
    }
    assert pitch_vocal[8000].confidence == 1.0

    # Tempo TODO

    # Sama
    sama = jam.search(namespace='segment_open')[0]['data']
    assert len(sama) == 52
    assert sama[0].time == 4.894
    assert sama[0].duration == 5.334999999999999
    assert sama[0].value == 'Sama cycle 1'
    assert sama[0].confidence is None

    assert sama[51].time == 301.959
    assert sama[51].duration == 5.812000000000012
    assert sama[51].value == 'Sama cycle 52'
    assert sama[51].confidence is None

    # Sections
    sections = jam.search(namespace='segment_open')[1]['data']
    assert [section.time for section in sections] == [
        0.065306122,
        85.35510204,
        167.314285714
    ]
    assert [section.duration for section in sections] == [
        85.289795918,
        81.95918367399999,
        142.02775510200001
    ]
    assert [section.value for section in sections] == [
        'Pallavi (1)',
        'Anupallavi (1)',
        'Caraṇam (1)'
    ]
    assert [section.confidence for section in sections] == [None, None, None]

    # Phrases
    phrases = jam.search(namespace='tag_open')[0]['data']
    assert [phrase.time for phrase in phrases] == [
        0.224489795, 5.844897959, 8.50430839, 16.734693877, 19.591836734, 30.918367346, 42.318367346,
        53.836734693, 62.450068027, 94.35510204, 100.265306122, 106.13877551, 109.171428571, 115.212244897,
        121.236734693, 126.995918367, 130.106122448, 139.763265306, 168.037006802, 173.789750566, 185.469387755,
        200.310204081, 206.342857142, 211.87755102, 217.6, 220.628571428, 226.310204081, 232.024489795, 238.195918367,
        243.85015873, 246.920997732, 255.523990929
    ]
    assert [phrase.duration for phrase in phrases] == [
        2.4938775509999997, 2.4734693870000006, 2.2755555550000004, 2.608163264999998, 2.408163264999999,
        2.542857141999999, 2.416326529999999, 2.3999999999999986, 3.9357823120000006, 2.767346938000003,
        2.8081632650000046, 2.8122448969999994, 2.873469387, 2.836734692999997, 2.767346938000003, 2.832653061000002,
        2.5510204079999994, 3.3836734690000014, 2.908299319000008, 2.8038095230000124, 2.844444444000004,
        3.0612244890000113, 2.5061224480000135, 2.8653061219999927, 2.9795918359999973, 2.824489795000005,
        3.1755102040000054, 2.685714284999989, 2.138775509999988, 2.6644897950000086, 2.693514738999994,
        4.400181404999984
    ]
    assert phrases[0].value == 'ndmdnsndn (0)'
    assert phrases[-1].value == 'ndmdnsndn (0)'

    assert [phrase.confidence for phrase in phrases] == [
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None
    ]

    # Metadata
    metadata = jam['sandbox'].metadata
    assert metadata['raaga'] == [{
        'uuid': '42dd0ccb-f92a-4622-ae5d-a3be571b4939',
        'name': 'Śrīranjani'
    }]
    assert metadata['form'] == [{
        'name': 'Kriti'
    }]
    assert metadata['title'] == 'Bhuvini Dasudane'
    assert metadata['work'] == [{
        'mbid': '4d05ce9b-c45e-4c85-9eca-941d68b61132',
        'title': 'Bhuvini Dasudane'
    }]
    assert metadata['length'] == 309000
    assert metadata['taala'] == [{
        'uuid': 'c788c38a-b53a-48cb-b7bf-d11769260c4d',
        'name': 'Ādi'
    }]
    assert metadata['album_artists'] == [{
        'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83',
        'name': 'Cherthala Ranganatha Sharma'
    }]
    assert metadata['mbid'] == '9f5a5452-14cb-4af0-9289-4833854ee60d'
    assert metadata ['artists'] == [
        {'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': '19f93366-5d58-47f1-bc4f-9225ac7af6ba', 'name': 'N Guruprasad'}},
        {'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': '39c1d741-6154-418b-bf4b-12c77ba13873', 'name': 'Srimushnam V Raja Rao'}},
        {'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': 'a2df55e3-d141-4767-862e-77adca691d4b', 'name': 'B.U. Ganesh Prasad'}},
        {'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'attributes': 'lead vocals',
         'lead': True,
         'artist': {'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}}
    ]
    assert metadata['concert'] == [{
        'mbid': '0816586d-c83e-4c79-a0aa-9b0e578f408d',
        'title': 'Cherthala Ranganatha Sharma at Arkay'
    }]
    assert metadata['track_id'] == 'carnatic_1'
    assert metadata['data_home'] == '/Users/genisplaja/Desktop/genis-datasets/saraga1.0'


def main():
    data_home = TEST_DATA_HOME
    # data_home = '../tests/resources/mir_datasets/Saraga/saraga1.0'
    ids = saraga.track_ids()
    data = saraga.load(data_home)

    track_carnatic = data['carnatic_1']
    test_to_jams()


if __name__ == '__main__':
    main()