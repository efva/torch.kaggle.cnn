import h5py
import glob
import json

imgs = glob.glob('/home/stathis/data/kaggle/imgs/test/*.jpg')

test = dict()
dict
test['image'] = []
test['name'] = []

for img in imgs:
    test['image'].append(img)
    test['name'].append(img.split('/')[-1])

json.dump(test,open('test.json','w'))

print test['image'][234]
print test['name'][234]