# -*- coding: utf-8 -*-
"""Saraga Dataset Loader

"""
import csv
import librosa
import logging
import numpy as np
import os
import json

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'Saraga'

REMOTES = {
    'all': download_utils.RemoteFileMetadata(
        filename='saraga_1.0.zip',
        url='https://zenodo.org/record/1256127/files/saraga_1.0.zip?download=1',
        checksum='c8471e55bd55e060bde6cfacc555e1b1',
        destination_dir=None,
    )
}


def _load_metadata(metadata_path):

    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    with open(metadata_path) as f:
        data = json.load(f)
        metadata = {}
        # Carnatic track
        if 'raaga' in data.keys():
            if data['raaga']:
                metadata['raaga'] = data['raaga'][0]['name']
            if data['work']:
                metadata['mbid'] = data['work'][0]['mbid']
                metadata['title'] = data['work'][0]['title']
            if data['album_artists']:
                metadata['artists'] = data['album_artists'][0]['name']

        # Hindustani tracks
        # if 'raags' in data.keys():
        # TODO

        data_home = metadata_path.split('/' + metadata_path.split('/')[-3])[0]
        metadata['data_home'] = data_home

        return metadata


DATA = utils.LargeData('saraga_index.json', _load_metadata)


class Track(track.Track):
    """salami Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:


    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Salami'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        # Annotation paths
        self.ctonic_path = utils.none_path_join(
            [self._data_home, self._track_paths['ctonic'][0]]
        )
        self.pitch_path = utils.none_path_join(
            [self._data_home, self._track_paths['pitch'][0]]
        )
        self.pitch_vocal_path = utils.none_path_join(
            [self._data_home, self._track_paths['pitch_vocal'][0]]
        )
        self.bpm_path = utils.none_path_join(
            [self._data_home, self._track_paths['bpm'][0]]
        )
        self.tempo_path = utils.none_path_join(
            [self._data_home, self._track_paths['tempo'][0]]
        )
        self.sama_path = utils.none_path_join(
            [self._data_home, self._track_paths['sama'][0]]
        )
        self.sections_path = utils.none_path_join(
            [self._data_home, self._track_paths['sections'][0]]
        )
        self.phrases_path = utils.none_path_join(
            [self._data_home, self._track_paths['phrases'][0]]
        )
        self.metadata_path = utils.none_path_join(
            [self._data_home, self._track_paths['metadata'][0]]
        )

        # Flag to separate between carnatinc and hindustani tracks
        self.style = str(self.track_id.split('_')[0])

        metadata = DATA.metadata(self.metadata_path)
        if metadata is not None and track_id in metadata.keys():
            self._track_metadata = metadata[track_id]
        else:
            # annotations with missing metadata
            self._track_metadata = {
                'raaga': None,
                'mbid': None,
                'title': None,
                'artists': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

    @utils.cached_property
    def tonic(self):
        """String: tonic annotation"""
        if self.ctonic_path is None:
            return None
        return load_tonic(self.ctonic_path)

    @utils.cached_property
    def pitch(self):
        """F0Data: pitch annotation"""
        if self.pitch_path is None:
            return None
        return load_pitch(self.pitch_path)

    @utils.cached_property
    def pitch_vocal(self):
        """F0Data: pitch vocal annotations"""
        if self.pitch_vocal_path is None:
            return None
        return load_pitch_vocal(self.pitch_vocal_path)

    @utils.cached_property
    def bpm(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.bpm_path is None:
            return None
        return load_bpm(self.bpm_path)

    @utils.cached_property
    def tempo(self):
        """TempoData: tempo annotations"""
        if self.tempo_path is None:
            return None
        return load_tempo(self.tempo_path)

    @utils.cached_property
    def sama(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.sama_path is None:
            return None
        return load_sama(self.sama_path)

    @utils.cached_property
    def sections(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.sections_path is None:
            return None
        return load_sections(self.sections_path)

    @utils.cached_property
    def phrases(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.phrases_path is None:
            return None
        return load_phrases(self.phrases_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata=self._track_metadata,
        )


def load_audio(audio_path):
    """Load a Saraga audio file.

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
    """Download Saraga Dataset (annotations).
    The audio files are not provided.

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

    info_message = """
        Unfortunately the audio files of the Salami dataset are not available
        for download. If you have the Salami dataset, place the contents into a
        folder called Salami with the following structure:
            > Salami/
                > salami-data-public-hierarchy-corrections/
                > audio/
        and copy the Salami folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=info_message,
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
    """Load Saraga dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in track_ids():
        data[key] = Track(key, data_home=data_home)
    return data


def load_tonic(sections_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """


def load_pitch(pitch_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """


def load_pitch_vocal(pitch_vocal_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """


def load_bpm(bpm_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """


def load_tempo(tempo_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """


def load_sama(sama_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """


def load_sections(sections_path):
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    times = []
    secs = []
    with open(sections_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            times.append(float(line[0]))
            secs.append(line[1])
    times = np.array(times)
    secs = np.array(secs)

    # remove sections with length == 0
    times_revised = np.delete(times, np.where(np.diff(times) == 0))
    secs_revised = np.delete(secs, np.where(np.diff(times) == 0))
    return utils.SectionData(
        np.array([times_revised[:-1], times_revised[1:]]).T, list(secs_revised[:-1])
    )


def load_phrases(phrases_path):
    """Load tonic

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    print('hola')


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Smith, Jordan Bennett Louis, et al.,
"Design and creation of a large-scale database of structural annotations",
12th International Society for Music Information Retrieval Conference (2011)

========== Bibtex ==========
@inproceedings{smith2011salami,
    title={Design and creation of a large-scale database of structural annotations.},
    author={Smith, Jordan Bennett Louis and Burgoyne, John Ashley and
          Fujinaga, Ichiro and De Roure, David and Downie, J Stephen},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2011},
    series = {ISMIR},
}
"""

    print(cite_data)


def main():
    data_home = '/Users/genisplaja/Desktop/genis-datasets/saraga1.0'
    ids = track_ids()
    data = load(data_home)

    example_track = data[ids[0]]
    print(example_track.metadata_path)
    metadata = _load_metadata(example_track.metadata_path)
    print(metadata)


if __name__ == '__main__':
    main()