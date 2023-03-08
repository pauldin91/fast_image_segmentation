import os 
import cv2
import natsort
import numpy as np

##############################################################################
path2ann = r'C:\Users\dipsa\Desktop\project\project_files\Updated_Pipeline_Dataset\annotations\val'

number_of_classes = 1        # BACKGORUND IS NOT INCLUDED
##############################################################################

def RGB2Ids(root_path):
    annotations = natsort.natsorted(os.listdir(root_path))
    for annotation in annotations:
        path = os.path.join(root_path,annotation)
        img = cv2.imread(path)
        
        Ids = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        classes = np.unique(Ids)
        
        for index, c in enumerate(classes):
            Ids[Ids==c] = index
            
        cv2.imwrite(os.path.join(root_path, annotation.split('.')[0] + 'Ids' + '.' + annotation.split('.')[1]),Ids)
    return

def Ids2RGB(root_path, classes):
    annotations = natsort.natsorted(os.listdir(root_path))
    colors = get_colors(classes)
    
    for annotation in annotations:
        path = os.path.join(root_path,annotation)
        Ids = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        rgb = np.zeros((Ids.shape[0],Ids.shape[1],3))
        classes = np.unique(Ids)[1:]
        
        for index, c in enumerate(classes):
            color = colors[index]
            rgb[np.where(Ids==c)[0],np.where(Ids==c)[1],0] = color[0]
            rgb[np.where(Ids==c)[0],np.where(Ids==c)[1],1] = color[1]
            rgb[np.where(Ids==c)[0],np.where(Ids==c)[1],2] = color[2]
        
        name = annotation.split('.')[0]
        rgb_name = name[:-3]
        rgb_name = rgb_name + '.' + annotation.split('.')[1]
        cv2.imwrite(os.path.join(root_path,rgb_name),rgb)
    return

def change_colors(root_path, classes):
    colors = get_colors(classes)
    annotations = natsort.natsorted(os.listdir(root_path))
    
    for annotation in annotations:
        path = os.path.join(root_path,annotation)
        img = cv2.imread(path)
        
        Ids = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        classes = np.unique(Ids)[1:]
        
        for index, c in enumerate(classes):
            color = colors[index]
            img[np.where(Ids==c)[0],np.where(Ids==c)[1],0] = color[0]
            img[np.where(Ids==c)[0],np.where(Ids==c)[1],1] = color[1]
            img[np.where(Ids==c)[0],np.where(Ids==c)[1],2] = color[2]
            
        cv2.imwrite(os.path.join(root_path, annotation.split('.')[0] + '.' + annotation.split('.')[1]),img)
    return

def get_colors(len_colors):
    seed = 13546
    np.random.seed(seed)
    
    colors = np.zeros((len_colors,3),np.ndarray)
    
    colors[0] = np.array([224,64,192])

    for i in range(len_colors-1):
        color = np.array(np.random.choice(range(50,256),size=3))
        colors[i+1] = color
        
    return colors


change_colors(path2ann, number_of_classes)
RGB2Ids(path2ann)
#RGB2Ids(path2ann)
#Ids2RGB(path2ann, number_of_classes)