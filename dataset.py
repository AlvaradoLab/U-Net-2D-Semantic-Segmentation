import glob
from torch.utils.data import Dataset

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

class FishDataset(Dataset):
    
    # folders: List of dataset folders with subfolders for body parts with naming convention as above.
    # type:
    #       full: whole_body semantic segmentation
    #       indep: independent parts finetuned using whole_body model
    #       ventral_side: assumes model for ventral_side already exists
    #       dorsal_side: assumes model for dorsal_side already exists
    #       head: assumes segmentation model for head already exists
    #       body: ventral_side, dorsal_side and head segmentations
    def __init__(self, folders, split='train', dtype='full'):
        
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
        
        self.img_files = []
        for folder in folders:
            
            imgs = [y for x in [glob.glob(e) for e in IMG_TYPES] for y in x]
            self.img_files.extend(imgs)
        
        print (self.img_files)

if __name__=='__main__':
    
    FishDataset(['/home/hans/Fish-Study/Machine learning training set/Light-Dark/T0', 
                 '/home/hans/Fish-Study/Machine learning training set/Light-Dark/T1',
                 '/home/hans/Fish-Study/Machine learning training set/photos 1.30.2019'], 'full')

    '''
    FishDataset('x', 'body')
    FishDataset('x', 'indep')
    FishDataset('x', 'ventral_side')
    FishDataset('x', 'dorsal_side')
    FishDataset('x', 'head')
    '''
