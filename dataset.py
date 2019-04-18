from pprint import pprint
import rawpy
import imageio
import glob
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.misc import imread

import time
random.seed(int(time.time()))

'''
HIERARCHY:
Ventral side -> pectoral anal fin
Dorsal side -> dorsal fin
Head -> Eye + Operculum

INDEPENDENT:
Whole body

Humeral blotch
Pelvic fin
Caudal Fin
'''

INIT = ['whole_body']
HPARTS = [['ventral_side', 'anal_fin', 'pectoral_fin'], ['dorsal_side', 'dorsal_fin'], ['head', 'eye', 'operculum']]
INDEP = ['humeral_blotch', 'pelvic_fin', 'caudal_fin']

IMG_TYPES = ['jpg', 'png', 'arw']
IMG_TYPES.extend([x.upper() for x in IMG_TYPES])

class FishDataset(Dataset):
    
    # folders: List of dataset folders with subfolders for body parts with naming convention as above.
    # type:
    #       full: whole_body semantic segmentation
    #       indep: independent parts finetuned using whole_body model
    #       ventral_side: assumes model for ventral_side already exists
    #       dorsal_side: assumes model for dorsal_side already exists
    #       head: assumes segmentation model for head already exists
    #       body: ventral_side, dorsal_side and head segmentations
    def __init__(self, folders, split='train', dtype='full', shuffle=False):
        
        self.split_ratio = 0.9 # train-val split
        
        if dtype=='full':
            self.dataset = INIT
        elif dtype=='indep':
            self.dataset = INDEP
        elif dtype=='body':
            self.dataset = [x[0] for x in HPARTS]
        else:
            for i in range(len(HPARTS)):
                if dtype==HPARTS[i][0]:
                    self.dataset = HPARTS[i][1:]
        
        self.shuffle = shuffle
        self.split = split

        # Dataset input images (x)
        self.img_files = {}
        # Dataset annotation images (y)
        self.ann_files = []
        for folder in folders:
            
            imgs = self.get_image_files(folder)
            self.img_files.update(imgs)
            
            for idx, fl in enumerate(self.dataset):
                dpath = os.path.join(folder, fl)
                anns = self.get_ann_files(dpath, self.img_files)

                if idx < len(self.ann_files):
                    self.ann_files[idx].update(anns)
                else:
                    self.ann_files.append(anns)
        
        N = len(list(self.img_files.keys()))
        n = int(N * self.split_ratio)
        
        num_del_imgs = N-n if split=='train' else n
        for _ in range(num_del_imgs):
            k = random.choice(list(self.img_files.keys()))
            del self.img_files[k]

            for ann in self.ann_files:
                del ann[k]
       
        self.ordered_keys = list(self.img_files.keys())

    def get_image_files(self, path):
        
        imgs = [y for x in [glob.glob(os.path.join(path, '*.'+e)) 
                  for e in IMG_TYPES] for y in x]

        img_dict = {}
        for path in imgs:
            sfx = '.'.join(path.split('/')[-1].split('.')[:-1])
            img_dict[sfx] = path

        return img_dict
    
    def get_segmentation_mask(self, path):
        
        ANN = self.get_image(path)
        gray = ANN.convert('L')
        bw = gray.point(lambda x: 0 if x==255 else 1, '1')
        
        #bw.save('sample.jpg')
        #print (np.max(np.array(bw))) 
        #print (np.min(np.array(bw))) 
        
        return bw

    def get_ann_files(self, path, imgs):
        
        ann_dict = {}
        for img_key in imgs:
            
            annfile = glob.glob(os.path.join(path, '*'+img_key+' *'))
            annfile.extend(glob.glob(os.path.join(path, '*'+img_key+'.*')))
            assert len(annfile) <= 1
            if annfile:
                ann_dict[img_key] = annfile[0]
                
                #img = self.get_image(imgs[img_key])
                #img.save('sample2.jpg')
                
        return ann_dict
    
    def __len__(self):
        return len(self.img_files)

    def get_image(self, path):
        
        if 'arw' in path.lower():
            raw = rawpy.imread(path)
            img = raw.postprocess()
            img = Image.fromarray(img)
        else:
            img = Image.open(path)

        return img

    def __getitem__(self, index):
        
        if self.shuffle:
            key = random.choice(self.ordered_keys)
        else:
            key = self.ordered_keys[index]
        
        image = self.get_image(self.img_files[key])

        # List of PIL Image objects
        anns = []
        for ann in self.ann_files:
            ann_img = self.get_image(ann[key])
            
            segmask = self.get_segmentation_mask(annfile[0]) 
            anns.append(segmask)
            
            #img2 = Image.fromarray(ANN)
            #img2.save('sample.jpg')
        
        return image, anns

        #if self.split != 'test':
        #    pass            
            

if __name__=='__main__':
    
    types = ['full', 'body', 'indep', 'ventral_side', 'dorsal_side', 'head']

    for t in types:
        print (t)
        f = FishDataset(['/home/hans/Haplochromis-Burtoni-Study/Machine learning training set/Light-Dark/T0', 
                         '/home/hans/Haplochromis-Burtoni-Study/Machine learning training set/Light-Dark/T1',
                         '/home/hans/Haplochromis-Burtoni-Study/Machine learning training set/photos 1.30.2019'], 
                         split='test', dtype=t)
        
        
    '''
    FishDataset('x', 'body')
    FishDataset('x', 'indep')
    FishDataset('x', 'ventral_side')
    FishDataset('x', 'dorsal_side')
    FishDataset('x', 'head')
    '''
