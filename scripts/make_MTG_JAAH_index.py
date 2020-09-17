import argparse
import hashlib
import json
import os


MTG_JAAH_INDEX_PATH = '../mirdata/indexes/MTG_JAAH_index.json'
BEATLES_ANNOTATION_SCHEMA = ['JAMS']


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_MTG_JAAH_index(data_path):
    annotations_dir = os.path.join(data_path, 'annotations')
    audio_dir = os.path.join(data_path, 'audio')
    chord_dir = os.path.join(data_path, 'labs')
    MTG_JAAH_index = {}
    for track_id, ann_dir in enumerate(sorted(os.listdir(annotations_dir))):
        ann_dir_full = os.path.join(annotations_dir, ann_dir)
        if '.json' in ann_dir:
            codec = '.flac' if ann_dir != 'blues_for_alice.json' else '.mp3'
            audio_path = os.path.join(audio_dir, ann_dir.replace('.json', codec))
            chord_path = os.path.join(chord_dir, ann_dir.replace('.json', '.lab'))
            MTG_JAAH_index[track_id] = {
                'audio': (audio_path, md5(audio_path)),
                'annotations': (ann_dir_full, md5(ann_dir_full)),
                'chordlabs': (chord_path, md5(chord_path)),
            }
    with open(MTG_JAAH_INDEX_PATH, 'w') as fhandle:
        json.dump(MTG_JAAH_index, fhandle, indent=2)


def main(args):
    make_MTG_JAAH_index(args.MTG_JAAH_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make MTG_JAAH index file.')
    PARSER.add_argument(
        'MTG_JAAH_data_path', type=str, help='Path to MTG_JAAH data folder.'
    )
    main(PARSER.parse_args())
