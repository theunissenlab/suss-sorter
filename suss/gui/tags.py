"""Customizable tags for units
"""

from enum import Enum

import suss.gui.config as config


class ClusterTag(Enum):
    SINGLEUNIT = 1
    MULTIUNIT = 2
    NOISY = 3
    STAR = 4
    AUDITORY = 5
    NONAUDITORY = 6


UserTag = Enum("UserTag", " ".join(config.USERS), module=__name__)
