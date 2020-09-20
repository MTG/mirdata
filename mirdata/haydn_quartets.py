# -*- coding: utf-8 -*-
"""Beatles Dataset Loader

The Haydn Quartets Dataset includes symbolic notation and chord annotations for the 6 "sun" haydn quartets
https://zenodo.org/record/1095630#.X2eTpi3FQUE

"""

import csv
import shutil

import librosa
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'haydn_quartets'
REMOTES = {
    '.': download_utils.RemoteFileMetadata(
        filename='haydn_op20_harm-1.1-alpha.zip',
        url='https://zenodo.org/record/1095630/files/haydn_op20_harm-1.1-alpha.zip?download=1',
        checksum='7be22d53d48d522550befe5b6369096e',
        destination_dir='.',
    )
}

DATA = utils.LargeData('haydn_quartets_index.json')


class Track(track.Track):
    """Beatles track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        symbolic_path (str): symbolic notation path
        annotation_path (str): humdrum path
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Beatles'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.symbolic_path = os.path.join(self._data_home, self._track_paths['original_score'][0])
        self.annotation_path = os.path.join(self._data_home, self._track_paths['annotations'][0])


    @utils.cached_property
    def humdrum(self):
        """ChordData: chord annotation"""
        pass
    @utils.cached_property
    def roman(self):
        """KeyData: key annotation"""
        pass

    @utils.cached_property
    def chords(self):
        """SectionData: section annotation"""
        pass

    @property
    def midi(self):
        """(np.ndarray, float): audio signal, sample rate"""
        pass

    def to_jams(self):
        """Jams: the track's data in jams format"""
        # return jams_utils.jams_converter(
        #     audio_path=self.audio_path,
        #     beat_data=[(self.humdrum, None)],
        #     section_data=[(self.midi, None)],
        #     chord_data=[(self.chords, None)],
        #     key_data=[(self.numerals, None)],
        #     metadata={'artist': 'The Beatles', 'title': self.title},
        # )
        pass


def load_audio(audio_path):
    """Load a Beatles audio file.

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
    """Download the Beatles Dataset (annotations).
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

    # use the default location: ~/mir_datasets/Beatles
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message="Haydn quartets downloaded.",
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )

    shutil.copytree(os.path.join(utils.get_default_dataset_path(DATASET_DIR), 'haydn_op20_harm-1.1-alpha', 'op20'),
                    os.path.join(utils.get_default_dataset_path(DATASET_DIR), 'op20'))


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
    """Load Beatles dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = Track(key, data_home=data_home)
    return beatles_data




def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========

Nestor Napoles Lopez. (2017). Joseph Haydn - String Quartets Op.20 - Harmonic Analysis Annotations Dataset (Version 
v1.1-alpha) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.1095630 

========== Bibtex ==========
@dataset{nestor_napoles_lopez_2017_1095630,
  author       = {Nestor Napoles Lopez},
  title        = {{Joseph Haydn - String Quartets Op.20 - Harmonic 
                   Analysis Annotations Dataset}},
  month        = dec,
  year         = 2017,
  publisher    = {Zenodo},
  version      = {v1.1-alpha},
  doi          = {10.5281/zenodo.1095630},
  url          = {https://doi.org/10.5281/zenodo.1095630}
}
    """

    print(cite_data)


if __name__ == '__main__':
    track_ids()
