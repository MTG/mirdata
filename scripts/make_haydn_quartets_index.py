import argparse
import hashlib
import json
import os


haydn_quartets_INDEX_PATH = '../mirdata/indexes/haydn_quartets_index.json'
BEATLES_ANNOTATION_SCHEMA = ['beat', 'chordlab', 'keylab', 'seglab']


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


def make_beatles_index(data_path):
    quartets_paths = [os.path.join(data_path, 'op20', str(n)) for n in range(1, 7)]
    movement_paths = [os.path.join(quartet_path, i) for quartet_path in quartets_paths for i in ['i', 'ii', 'iii', 'iv']]
    movements = ['op20n' + str(number) + '-0' + str(movement) for number in range(1, 7) for movement in range(1, 5)]
    beatles_index = {}
    track_id = 0
    for p, m in zip(movement_paths, movements):
        original_score = os.path.join(p, m + '.krn')
        annotations = os.path.join(p, m + '.hrm')
        beatles_index[track_id] = {
            'original_score': (original_score.replace(data_path + '/', ''), md5(original_score)),
            'annotations': (annotations.replace(data_path + '/', ''), md5(annotations))
        }
        track_id += 1
    with open(haydn_quartets_INDEX_PATH, 'w') as fhandle:
        json.dump(beatles_index, fhandle, indent=2)


def main(args):
    make_beatles_index(args.beatles_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make Beatles index file.')
    PARSER.add_argument(
        'beatles_data_path', type=str, help='Path to Beatles data folder.'
    )

    main(PARSER.parse_args())
