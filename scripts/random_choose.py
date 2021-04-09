import random
import shutil
import os
from os import listdir
from os.path import isfile, join

getfiles = [f for f in listdir('./1_articles/') if isfile(join('./1_articles/', f))]
filenames = random.sample(getfiles, 31)

for filename in filenames:
    shutil.move('./1_articles/' + filename, './2_annotated_articles/' + filename)
