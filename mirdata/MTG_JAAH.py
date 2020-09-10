# -*- coding: utf-8 -*-
"""MTG-JAAH Dataset Loader

The MTG-JAAH Dataset includes beat and metric position, mbid, metre, tuning chord, key, and segmentation
annotations for 113 jazz songs. Details can be found in https://github.com/MTG/JAAH .

"""

import csv
import json
import math
import re

import librosa
import numpy as np
import os
import shutil

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils
import scripts.pychord_tools_utils as chords_utils

DATASET_DIR = 'MTG_JAAH'
REMOTES = {
    'annotations': download_utils.RemoteFileMetadata(
        filename='JAAH-v0.1.zip',
        url='https://zenodo.org/record/1290737/files/MTG/JAAH-v0.1.zip?download=1',
        checksum='34f8311a270b9934cf2c9c0d6026ac71',
        destination_dir='',
    )
}

DATA = utils.LargeData('MTG_JAAH_index.json')


class Track(track.Track):
    """MTG-JAAH track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): track audio path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in MTG-JAAH'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.ann_path = os.path.join(self._data_home, self._track_paths['annotations'][0])
        with open(self.ann_path) as json_file:
            self.ann = json.load(json_file)
        # esto no se si va
        self.title = self.ann['title']

    @utils.cached_property
    def beats(self):
        """BeatData: human-labeled beat annotation"""
        beat_times = []
        for part in self.ann['parts']:
            beat_times += part['beats']
        metre = int(self.metre.split('/')[0])
        beat_positions = [p % metre + 1 for p in range(len(beat_times))]
        print(beat_positions)
        beat_data = utils.BeatData(np.array(beat_times), np.array(beat_positions))
        return beat_data

    @utils.cached_property
    def chords(self):
        """ChordData: chord annotation"""
        data = self.ann
        duration = float(data['duration'])
        all_beats = []
        all_chords = []
        chords_utils.process_parts(data['metre'], data, all_beats, all_chords, 'chords')
        segments = chords_utils.merge_segments(
            chords_utils.to_beat_chord_segment_list(0, duration, all_beats, all_chords))
        start_times = []
        end_times = []
        chords = []
        for s in segments:
            start_times.append(s.start_time)
            end_times.append(s.end_time)
            chords.append(s.symbol)
        chord_data = utils.ChordData(np.array([start_times, end_times]).T, chords)
        return chord_data

    @utils.cached_property
    def key(self):
        """KeyData: key annotation"""
        return self.ann['sandbox']['key']

    @utils.cached_property
    def artist(self):
        """title: title annotation"""
        return self.ann['artist']

    @utils.cached_property
    def tuning(self):
        """tuning: tuning annotation"""
        return self.ann['tuning']

    @utils.cached_property
    def metre(self):
        """metre: metre annotation"""
        return self.ann['metre']

    @utils.cached_property
    def mbid(self):
        """mbid: mbid annotation"""
        return self.ann['mbid']

    @utils.cached_property
    def duration(self):
        """duration: duration annotation"""
        return self.ann['duration']

    @utils.cached_property
    def sections(self):
        """SectionData: section annotation"""
        start_times = []
        end_times = []
        sections = []
        for part in self.ann['parts']:
            start_times.append(part['beats'][0])
            end_times.append(part['beats'][-1])
            sections.append(part['name'])
        print(start_times, end_times, sections)
        section_data = utils.SectionData(np.array([start_times, end_times]).T, sections)

        return section_data

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return self.ann


def load_audio(audio_path):
    """Load a MTG-JAAH audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download the MTG-JAAH Dataset (annotations).
    The audio files are not provided due to copyright issues.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """

    # use the default location: ~/mir_datasets/MTG_JAAH
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = """
        Unfortunately the audio files of the MTG_JAAH dataset are not available
        for download. If you have the MTG-JAAH dataset, place the contents into
        a folder called MTG-JAAH with the following structure:
            > MTG_JAAH/
                > annotations/
                > audio/
        and copy the MTG_JAAH folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=download_message,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )
    shutil.move(os.path.join(utils.get_default_dataset_path(DATASET_DIR), 'MTG-JAAH-7686b91', 'annotations'),
                utils.get_default_dataset_path(DATASET_DIR))


def validate(data_home=None, silence=False):
    """Validate if a local version of this dataset is consistent

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths where the expected file exists locally
            but has a different checksum than the reference

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Get the list of track IDs for this dataset

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load MTG_JAAH dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    MTG_JAAH = {}
    for key in track_ids():
        MTG_JAAH[key] = Track(key, data_home=data_home)
    return MTG_JAAH


def _fix_newpoint(beat_positions):
    """ Fills in missing beat position labels by inferring the beat position
        from neighboring beats.

    """
    while np.any(beat_positions == 'New Point'):
        idxs = np.where(beat_positions == 'New Point')[0]
        for i in idxs:
            if i < len(beat_positions) - 1:
                if not beat_positions[i + 1] == 'New Point':
                    beat_positions[i] = str(np.mod(int(beat_positions[i + 1]) - 1, 4))
            if i == len(beat_positions) - 1:
                if not beat_positions[i - 1] == 'New Point':
                    beat_positions[i] = str(np.mod(int(beat_positions[i - 1]) + 1, 4))
    beat_positions[beat_positions == '0'] = '4'

    return beat_positions


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========

Eremenko, V., Demirel, E., Bozkurt, B., & Serra, X. (2018). 
Audio-aligned jazz harmony dataset for automatic chord transcription and corpus-based research. 
International Society for Music Information Retrieval Conference.

========== Bibtex ==========
@conference {3896,
    title = {Audio-aligned jazz harmony dataset for automatic chord transcription and corpus-based research},
    booktitle = {International Society for Music Information Retrieval Conference},
    year = {2018},
    month = {23/09/2018},
    address = {Paris},
    url = {https://doi.org/10.5281/zenodo.1291834},
    author = {Eremenko, Vsevolod and Demirel, Emir and Bozkurt, Bar{\i}{\c s} and Xavier Serra}
}
    """

    print(cite_data)


if __name__ == '__main__':
    data = load()
    data['1'].chords()
