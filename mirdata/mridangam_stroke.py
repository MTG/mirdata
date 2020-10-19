#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Mridangam Stroke Dataset Loader

The Mridangam Stroke dataset is a collection of individual strokes of
the Mridangam in various tonics. The dataset comprises of 10 different
strokes played on Mridangams with 6 different tonic values. The audio
examples were recorded from a professional Carnatic percussionist in a
semi-anechoic studio conditions by Akshay Anantapadmanabhan.

Total audio samples: 6977

Used microphones:
* SM-58 microphones
* H4n ZOOM recorder.

Audio specifications
* Sampling frequency: 44.1 kHz
* Bit-depth: 16 bit
* Audio format: .wav

The dataset can be used for training models for each Mridangam stroke. The
presentation of the dataset took place on the IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP 2013) on May 2013.
You can read the full publication here: https://repositori.upf.edu/handle/10230/25756

Mridangam Dataset is annotated by storing the informat of each track in their filenames.
The structure of the filename is:
<TrackID>__<AuthorName>__<StrokeName>-<Tonic>-<InstanceNum>.wav

The dataset is made available by CompMusic under a Creative Commons
Attribution 3.0 Unported (CC BY 3.0) License.

For more details, please visit: https://compmusic.upf.edu/mridangam-stroke-dataset
"""

import os
import librosa

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils


DATASET_DIR = 'Mridangam-Stroke'


REMOTES = {
    'remote_data': download_utils.RemoteFileMetadata(
        filename='mridangam_stroke_1.5.zip',
        url='https://zenodo.org/record/4068196/files/mridangam_stroke_1.5.zip?download=1',
        checksum='39af55b2476b94c7946bec24331ec01a',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
}

DATA = utils.LargeData(
    'mridangam_stroke_index.json'
)  # use this if your dataset has no metadata


class Track(track.Track):
    """Mridangam Stroke track class
    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Mridangam-Stroke`
    Attributes:
        track_id (str): track id
        stroke_name (str): stroke name or type
        tonic (str): tonic note of the stroke
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Example'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    @property
    def stroke_name(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_stroke_name(self.audio_path)

    @property
    def tonic(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_tonic(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            tags_open_data=[(self.stroke_name, 'stroke_name')],
            metadata={'tonic': self.tonic},
        )


def load_audio(audio_path):
    """Load a Mridangam Stroke Dataset audio file.
    Args:
        audio_path (str): path to audio file
    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=44100, mono=True)


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download the Mridangam Stroke Dataset.
    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=None,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version
    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids
    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load Mridangam Stroke dataset
    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in DATA.index.keys():
        data[key] = Track(key, data_home=data_home)
    return data


def load_stroke_name(audio_path):
    """Load stroke name of track
    Args:
        audio_path (str): Local path where the track is stored.
    Returns:
        stroke_name (str): stroke name or type extracted from filename
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    if '.wav' in audio_path:
        audio_data = audio_path.split('__')[2]
        stroke_name = audio_data.split('-')[0]

        return stroke_name


def load_tonic(audio_path):
    """Get tonic of track
    Args:
        audio_path (str): Local path where the track is stored.
    Returns:
        tonic (str): track tonic extracted from filename
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    if '.wav' in audio_path:
        audio_data = audio_path.split('__')[2]
        tonic_info = audio_data.split('-')[1]
        tonic_info_upper = tonic_info.upper()

        # Adapt sharp tonic: 'sh' to '#'
        if 'SH' in tonic_info_upper:
            tonic_info_upper = tonic_info_upper[0] + '#'

        return tonic_info_upper


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Anantapadmanabhan, A., Bellur, A., & Murthy, H. A. 
"Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization" (2013)
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2013)
========== Bibtex ==========
@article{Anantapadmanabhan2013,
    author = {Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A.},
    doi = {10.1109/ICASSP.2013.6637633},
    isbn = {9781479903566},
    issn = {15206149},
    journal = {ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings},
    keywords = {Hidden Markov models, Modal Analysis, Mridangam, Non-negative Matrix Factorization,
    automatic transcription},
    pages = {181--185},
    title = {{Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization}},
    year = {2013}
}
"""
    print(cite_data)