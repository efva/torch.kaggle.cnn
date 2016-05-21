import h5py
import glob
import json
import numpy as np


ALL_DRIVERS =np.array(['p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                       'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                       'p050',  'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                       'p075', 'p081', 'p051', 'p002'])
   
def get_driver_data():
    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index

def select_drivers( filenames, driver_list, info):
    ids = []
    for i in range(len(filenames)):    
        if info[ filenames[i] ] in driver_list:
            ids.append(i)
    return ids

split = dict()
split["train"] = [ 0, 10,  9, 18, 23,  8, 13,  4,  7, 20,  5, 16, 17,  3, 25, 12, 11, 19,  1, 15],
split["valid"] = [22, 24, 21, 14, 2,  6]    
unique_list_train = ALL_DRIVERS[split['train']]
unique_list_valid = ALL_DRIVERS[split['valid']]

info = get_driver_data()

imagePath = []
imageClass = []
names = []

for i in range(10):
    imgs = glob.glob('/home/stathis/data/kaggle/imgs/train/c'+str(i)+'/*.jpg')
    imagePath.extend(imgs)
    imageClass.extend([i]*len(imgs))
    for name in imgs:
        names.append(name.split('/')[-1])


train_index = select_drivers(names, unique_list_train, info)
valid_index = select_drivers(names, unique_list_valid, info)

imagePath = np.array(imagePath)
imageClass = np.array(imageClass)

dbfile = dict()
dbfile['basedir'] = ''
dbfile['classList'] = ['c0','c1','c2','c3','c4','c5', 'c6', 'c7', 'c8', 'c9']
dbfile['train'] = dict()
dbfile['train']['imagePath'] = imagePath[train_index]
dbfile['train']['imageClass'] = imageClass[train_index]
dbfile['val'] = dict()
dbfile['val']['imagePath'] = imagePath[valid_index]
dbfile['val']['imageClass'] = imageClass[valid_index]


json.dump(dbfile,open('split.json','w'))
