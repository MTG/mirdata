# -*- coding: utf-8 -*-
"""MTG-JAAH Dataset Loader

The MTG-JAAH Dataset includes beat and metric position, mbid, metre, tuning chord, key, and segmentation
annotations for 113 jazz songs. Details can be found in https://github.com/MTG/JAAH .

"""
import csv
import json
from zipfile import ZipFile

import librosa
import numpy as np
import os
import shutil

from mirdata import download_utils

from mirdata import track
from mirdata import utils

DATASET_DIR = 'MTG_JAAH'
REMOTES = {
    'annotations': download_utils.RemoteFileMetadata(
        filename='MTG-JAAH.zip',
        url='https://github.com/MTG/JAAH/archive/fb60f8a5bc3af692aa530dec5654ca9cebc6f63b.zip',
        destination_dir='.',
        checksum='7b728446516ad6e19532eeee51398a7e'
    ),
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
        self.labs_path = os.path.join(self._data_home, self._track_paths['chordlabs'][0])
        self.ann_path = os.path.join(self._data_home, self._track_paths['annotations'][0])

    @utils.cached_property
    def beats(self):
        return load_beats(self.ann_path)

    @utils.cached_property
    def chords(self):
        return load_chords(self.labs_path)

    @utils.cached_property
    def sections(self):
        """SectionData: section annotation"""
        return load_sections(self.ann_path)

    @utils.cached_property
    def key(self):
        """KeyData: key annotation"""
        return load_key(self.ann_path)

    @utils.cached_property
    def artist(self):
        """title: title annotation"""
        return load_artist(self.ann_path)

    @utils.cached_property
    def tuning(self):
        """tuning: tuning annotation"""
        return load_tuning(self.ann_path)

    @utils.cached_property
    def metre(self):
        """metre: metre annotation"""
        return load_metre(self.ann_path)

    @utils.cached_property
    def mbid(self):
        """mbid: mbid annotation"""
        return load_mbid(self.ann_path)

    @utils.cached_property
    def duration(self):
        """duration: duration annotation"""
        return load_duration(self.ann_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return self.ann


def load_beats(path):
    """Load MTG_JAAH format beat data from a file

    Args:
        path (str): path to beat annotation file

    Returns:
        (utils.BeatData): loaded beat data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    beat_times = []
    for part in ann['parts']:
        beat_times += part['beats']
    metre = int(ann['metre'].split('/')[0])
    beat_positions = [p % metre + 1 for p in range(len(beat_times))]

    beat_data = utils.BeatData(np.array(beat_times), np.array(beat_positions))
    return beat_data


def load_chords(path):
    """Load MTG_JAAH format chord data from a file

    Args:
        path (str): path to chord annotation file

    Returns:
        (utils.ChordData): loaded chord data

    """
    start_times = []
    end_times = []
    chords = []
    delimeter = '\t'
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimeter)
        for row in csv_reader:
            start_times.append(row[0])
            end_times.append(row[1])
            chords.append(row[2])
    return utils.ChordData(np.array([start_times, end_times]).T, chords)


def load_sections(path):
    """Load MTG_JAAH format section data from a file

        Args:
            path (str): path to section annotation file

        Returns:
            (utils.SectionData): loaded section data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    start_times = []
    end_times = []
    sections = []
    for part in ann['parts']:
        start_times.append(part['beats'][0])
        end_times.append(part['beats'][-1])
        sections.append(part['name'])
    print(start_times, end_times, sections)
    section_data = utils.SectionData(np.array([start_times, end_times]).T, sections)
    return section_data


def load_key(path):
    """Load MTG_JAAH format key data from a file

    Args:
        path (str): path to key annotation file

    Returns:
        (str): loaded key data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    return ann['sandbox']['key']


def load_artist(path):
    """Load MTG_JAAH format artist data from a file

    Args:
        path (str): path to artist annotation file

    Returns:
        (str): loaded artist data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    return ann['artist']


def load_tuning(path):
    """Load MTG_JAAH format tuning data from a file

    Args:
        path (str): path to tuning annotation file

    Returns:
        (float): loaded tuning data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    return ann['tuning']


def load_metre(path):
    """Load MTG_JAAH format metre data from a file

    Args:
        path (str): path to metre annotation file

    Returns:
        (str): loaded metre data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    return ann['metre']


def load_mbid(path):
    """Load MTG_JAAH format mbid data from a file

    Args:
        path (str): path to mbid annotation file

    Returns:
        (str): loaded mbid data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    return ann['mbid']


def load_duration(path):
    """Load MTG_JAAH format duration data from a file

    Args:
        path (str): path to duration annotation file

    Returns:
        (float): loaded duration data

    """
    ann = None
    with open(path) as json_file:
        ann = json.load(json_file)
    return ann['duration']


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
                > chordlab/
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
    ann = os.path.join(utils.get_default_dataset_path(DATASET_DIR), 'JAAH-fb60f8a5bc3af692aa530dec5654ca9cebc6f63b',
                       'annotations')
    labs = os.path.join(utils.get_default_dataset_path(DATASET_DIR), 'JAAH-fb60f8a5bc3af692aa530dec5654ca9cebc6f63b',
                        'labs.zip')
    shutil.copytree(ann, os.path.join(utils.get_default_dataset_path(DATASET_DIR), 'annotations'))
    print(labs, utils.get_default_dataset_path(DATASET_DIR))
    with ZipFile(labs, 'r') as zip_ref:
        zip_ref.extractall(utils.get_default_dataset_path(DATASET_DIR))


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
    # download()
    dataset = load()
    print(dataset['0'].artist)

    # for key, value in dataset.items():
    #     print(key)
    #     ans = value.beats




