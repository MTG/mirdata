# -*- coding: utf-8 -*-

from .version import version as __version__

DATASETS = [
    "beatles",
    "beatport_key",
    "cante100",
    "dali",
    "giantsteps_key",
    "giantsteps_tempo",
    "groove_midi",
    "gtzan_genre",
    "guitarset",
    "ikala",
    "irmas",
    "maestro",
    "medley_solos_db",
    "medleydb_melody",
    "medleydb_pitch",
    "mridangam_stroke",
    "orchset",
    "rwc_classical",
    "rwc_jazz",
    "rwc_popular",
    "salami",
    "saraga",
    "tinysol",
]

from .core import Dataset
