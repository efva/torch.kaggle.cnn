import h5py
import glob
import json
import numpy as np

split = dict()
dict
split['train'] = []
split['trainy'] = []
split['val'] = []
split['valy'] = []

valid_num = 400

for i in range(10):
    imgs = glob.glob('/home/stathis/data/kaggle/imgs/train/c'+str(i)+'/*.jpg')
    imgs = np.array(imgs)
    np.random.shuffle(imgs)
    split['val'].extend(imgs[:valid_num])
    split['valy'].extend([i+1]*valid_num )
    split['train'].extend(imgs[valid_num:])
    split['trainy'].extend([i+1] * len(imgs[valid_num:]))

split = dict()
dict
split['train'] = []
split['trainy'] = []
split['val'] = []
split['valy'] = []

valid_num = 400

for i in range(10):
    imgs = glob.glob('/home/stathis/data/kaggle/imgs/train/c'+str(i)+'/*.jpg')
    imgs = np.array(imgs)
    np.random.shuffle(imgs)
    split['val'].extend(imgs[:valid_num])
    split['valy'].extend([i+1]*valid_num )
    split['train'].extend(imgs[valid_num:])
    split['trainy'].extend([i+1] * len(imgs[valid_num:]))

json.dump(dataset,open('split.json','w'))
