#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
cante100 Loader

TODO FIX DESCRIPTION

The Mridangam Stroke dataset is a collection of individual strokes of
the Mridangam in various tonics. The dataset comprises of 10 different
strokes played on Mridangams with 6 different tonic values. The audio
examples were recorded from a professional Carnatic percussionist in a
semi-anechoic studio conditions by Akshay Anantapadmanabhan.

Total audio samples: 7162

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

import logging
import os
import csv
import numpy as np
import librosa
import xml.etree.ElementTree as ET

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils


DATASET_DIR = 'cante100'


REMOTES = {
    'spectrum': download_utils.RemoteFileMetadata(
        filename='cante100_spectrum.zip',
        url='https://zenodo.org/record/1322542/files/cante100_spectrum.zip?download=1',
        checksum='0b81fe0fd7ab2c1adc1ad789edb12981',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
    'melody': download_utils.RemoteFileMetadata(
        filename='cante100midi_f0.zip',
        url='https://zenodo.org/record/1322542/files/cante100midi_f0.zip?download=1',
        checksum='cce543b5125eda5a984347b55fdcd5e8',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
    'notes': download_utils.RemoteFileMetadata(
        filename='cante100_automaticTranscription.zip',
        url='https://zenodo.org/record/1322542/files/cante100_automaticTranscription.zip?download=1',
        checksum='47fea64c744f9fe678ae5642a8f0ee8e',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
}


def _load_metadata(data_home):
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    metadata_path = os.path.join(data_home, 'cante100Meta.xml')
    if not os.path.exists(metadata_path):
        logging.info(
            'Metadata file {} not found.'.format(metadata_path)
            + 'You can download the metadata file for cante100 '
            + 'by running cante100.download()'
        )
        return None

    tree = ET.parse(metadata_path)
    root = tree.getroot()

    # ids
    indexes = []
    for child in root:
        index = child.attrib.get('id')
        if len(index) == 1:
            index = '00' + index
            indexes.append(index)
            continue
        if len(index) == 2:
            index = '0' + index
            indexes.append(index)
            continue
        else:
            indexes.append(index)

    # musicBrainzID
    identifiers = []
    for ident in root.iter('musicBrainzID'):
        identifiers.append(ident.text)

    # artist
    artists = []
    for artist in root.iter('artist'):
        artists.append(artist.text)

    # titles
    titles = []
    for title in root.iter('title'):
        titles.append(title.text)

    # releases
    releases = []
    for release in root.iter('anthology'):
        releases.append(release.text)

    # duration
    durations = []
    minutes = []
    for minute in root.iter('duration_m'):
        minutes.append(int(minute.text) * 60)
    seconds = []
    for second in root.iter('duration_s'):
        seconds.append(int(second.text))
    for i in np.arange(len(minutes)):
        durations.append(minutes[i] + seconds[i])

    metadata = dict()
    metadata['data_home'] = data_home
    j = 0
    for i in indexes:
        metadata[i] = {
            'musicBrainzID': identifiers[j],
            'artist': artists[j],
            'title': titles[j],
            'release': releases[j],
            'duration': durations[j],
        }
        j += 1

    return metadata


DATA = utils.LargeData('cante100_index.json', _load_metadata)


class Track(track.Track):
    """cante100 track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Mridangam-Stroke`

    Attributes:
        track_id (str): track id
        # -- Add any of the dataset specific attributes here

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Example'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home

        # -- add any dataset specific attributes here
        self._track_paths = DATA.index[track_id]
        self.spectrum_path = os.path.join(
            self._data_home, self._track_paths['spectrum'][0]
        )
        self.f0_path = os.path.join(self._data_home, self._track_paths['f0'][0])
        self.notes_path = os.path.join(self._data_home, self._track_paths['notes'][0])

        metadata = DATA.metadata(data_home=data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                'musicBrainzID': None,
                'artists': None,
                'title': None,
                'release': None,
                'duration': None,
            }

        """
        self.identifier = self._track_metadata['musicBrainzID']
        self.artist = self._track_metadata['artist']
        self.title = self._track_metadata['title']
        self.release = self._track_metadata['release']
        self.duration = self._track_metadata['duration']
        """

    @property
    def spectrum(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_spectrum(self.spectrum_path)

    @property
    def melody(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_melody(self.f0_path)

    @property
    def notes(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_notes(self.notes_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            spectrum_cante100_path=self.spectrum_path,
            f0_data=[(self.melody, 'pitch_contour')],
            note_data=[(self.notes, 'note_hz')],
            metadata=self._track_metadata,
        )


def load_spectrum(spectrum_path):
    """Load a cante100 dataset audio file.

    Args:
        spectrum_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(spectrum_path):
        raise IOError("audio_path {} does not exist".format(spectrum_path))

    with open(spectrum_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='\n')

        spectrum_ = []
        total_rows = 0
        for row in reader:
            spectrum_.append(row[:514])
            total_rows += 1

        spectrum = np.array(spectrum_)
        spectrum.reshape(total_rows, 514)

    return spectrum


def load_melody(f0_path):
    """Load a cante100 dataset audio file.

    Args:
        f0_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(f0_path):
        raise IOError("audio_path {} does not exist".format(f0_path))

    times = []
    freqs = []
    with open(f0_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\n')
        for row in reader:
            times.append(float(row[0]))
            freqs.append(float(row[1]))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)

    melody_data = utils.F0Data(times, freqs, confidence)

    return melody_data


def load_notes(notes_path):
    """Load note data from the midi file.

    Args:
        notes_path (str): path to notes file

    Returns:
        note_data (NoteData)

    """
    if not os.path.exists(notes_path):
        raise IOError("audio_path {} does not exist".format(notes_path))

    intervals = []
    pitches = []
    confidence = []
    with open(notes_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\n')
        for row in reader:
            intervals.append([row[0], float(row[0]) + float(row[1])])
            pitches.append((440 / 32) * (2 ** ((int(row[2]) - 9) / 12)))
            confidence.append(1.0)

    note_data = utils.NoteData(
        np.array(intervals, dtype='float'),
        np.array(pitches, dtype='float'),
        np.array(confidence, dtype='float'),
    )

    return note_data


def download(
    data_home=None, partial_download=None, force_overwrite=False, cleanup=True
):
    """Download the cante100 dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        partial_download (str):
            TODO
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message='TODO',
        partial_download=partial_download,
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
    """Load cante100 dataset

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


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========

========== Bibtex ==========

"""

    print(cite_data)
