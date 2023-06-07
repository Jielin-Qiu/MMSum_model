from utils import open_file
import glob
import numpy as np

list_of_features = glob.glob('../../../A2Summ/data/MSMO/feature/*')
list_of_video_features = [list_of_features[1], list_of_features[3]]
list_of_keyframe_features = [list_of_features[2], list_of_features[5]]
list_of_features = list_of_video_features + list_of_keyframe_features


for path in list_of_features:
    if path == '../../../A2Summ/data/MSMO/feature/msmo_clip_features_test.npy' or \
        path == '../../../A2Summ/data/MSMO/feature/msmo_clip_features_train.npy':
        dic = open_file(path)
        
        for key in dic.keys():
            arr = dic[f'{key}']
            # np.save(f'./videos/{key}.npy', arr)

    else:
        dic = open_file(path)
        for key in dic.keys():
            arr = dic[f'{key}']
            # np.save(f'./keyframes/{key}.npy', arr)

    
    
            