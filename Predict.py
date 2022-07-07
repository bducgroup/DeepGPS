from Model import KingsleyModel as DeepGPS
from InputDataset import *
from tqdm import tqdm
from math import floor,sqrt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

# Load Deepgps dataset
src_path = "./GPS_sample/"
ds_origin = PreDataset(src_path)
size_train = int(len(ds_origin)*0)
size_valid = int(len(ds_origin)) - size_train
ds_train,ds_valid = random_split(ds_origin,[size_train,size_valid])
dl_test = DataLoader(ds_valid, batch_size=32,shuffle=False)
print('len:',len(dl_test))

# load Deepgps model
model = DeepGPS().cuda()
model_path = "./trained_model.pkl"
print('model_path',model_path)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['pos_model'])
model.eval()


ob_gt = []
cor_gt = []
filename_arr = []
point_cnt = 0
for step, (features, labels) in tqdm(enumerate(dl_test, 1)):
    # feed environment matrix, timestamp matrix and sky matrix into model
    x1, x2,x3 = features[0], features[1], features[2]
    predictions = model(x1,x2,x3)

    # get the element with the largest probablity and get the element of ground truth
    predictions = predictions.squeeze().cpu().detach().numpy()
    filenames = labels[2]
    labels = labels[0].squeeze().cpu().detach().numpy()
    predictions = np.array([np.unravel_index(i.argmax(), predictions.shape) for i in predictions])
    labels = np.array([np.unravel_index(i.argmax(), labels.shape) for i in labels])


    for idx,label in enumerate(zip(labels,filenames)):
        label,filename = label

        gt_x,gt_y = int(label[1]),int(label[2])
        point = predictions[idx]
        pre_x1,pre_y1 = floor(float(point[1])+0.5),floor(float(point[2])+0.5)
        point_cnt+=1

        print('========='*5)
        print(point_cnt)
        print(filename)
        print('x1,y1,x2,y2:',gt_x,gt_y,pre_x1,pre_y1)
        print('Accuracy before correction:',sqrt((50-gt_y)**2+(50-gt_x)**2))
        print('Accuracy after correction:',sqrt((pre_y1-gt_y)**2+(pre_x1-gt_x)**2))
        print()

