'''
데이터 분리하기(train / test)
'''

import os
import random
import numpy as np 
import shutil

random.seed(1234)

idx = random.sample(range(853), 170)

if not os.path.isdir('data/test_images'):
    os.mkdir('data/test_images')
    os.mkdir('data/test_annotations')

    for img in np.array(sorted(os.listdir('data/images')))[idx]:
        shutil.move('data/images/'+ img, 'data/test_images/' + img)

    for annot in np.array(sorted(os.listdir('data/annotations')))[idx]:
        shutil.move('data/annotations/'+ annot, 'data/test_annotations/' + annot)

# print(len(os.listdir('data/annotations')))
# print(len(os.listdir('data/images')))
# print(len(os.listdir('data/test_annotations')))
# print(len(os.listdir('data/test_images')))

print("Split data")