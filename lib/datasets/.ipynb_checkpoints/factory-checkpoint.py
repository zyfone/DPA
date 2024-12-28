# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function
from cgi import test

import numpy as np
from datasets.clipart import clipart
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.water import water

from datasets.openset_clipart import openset_clipart
from datasets.openset_clipart_test import openset_clipart_test
from datasets.openset_voc import openset_voc

from datasets.openset_water import openset_water
__sets = {}



for altha in ["0.75","0.5","0.25"]:
    for split in ["train"]:
            name = "clipart_{}_{}".format(split,altha)
            __sets[name] = lambda split=split,altha=altha: openset_clipart(image_set=split, altha=altha)

for altha in ["0.75","0.5","0.25"]:
    for split in ["test"]:
            name = "clipart_{}_{}".format(split,altha)
            __sets[name] = lambda split=split,altha=altha: openset_clipart_test(image_set=split, altha=altha)


for altha in ["0.75","0.5","0.25","1.0"]:
    for year in ["2007", "2012"]:
        for split in ["trainval"]:
            name = "voc_{}_{}_{}".format(year, split,altha)
            __sets[name] = lambda split=split, year=year,altha=altha: openset_voc(image_set=split, altha=altha, year=year)



for year in ["2007"]:
    for split in ["test"]:
        name = "voc_water_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_water(split, year)


for year in ["2007"]:
    for split in ["train", "test"]:
        name = "water_{}".format(split)
        __sets[name] = lambda split=split: water(split, year)


for year in ["2007"]:
    for split in ["train","test"]:
        name = "water_openset_{}".format(split)
        __sets[name] = lambda split=split: openset_water(split, year)



def get_imdb(name):

    """Get an imdb (image database) by name."""

    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
