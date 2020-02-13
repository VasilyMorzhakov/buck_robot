import torch.optim as optim
from torch import nn
from feature_detector import model
from feature_detector import prepair_data
import config
import random
import numpy
import torch
import cv2

batch_size=16

model=model.Detector(config.num_features).cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

N=1000
data_input=numpy.zeros((N,3,config.detector_input_W,config.detector_input_W))
data_output=numpy.zeros((N,config.num_features,config.detector_output_W,config.detector_output_W))
for i in range(N):
    feature_index=random.randint(0,config.num_features-1)
    frame_index = random.randint(0, config.num_frames_for_features - 1)
    scale=random.random()+0.5
    alpha=random.random()*60.0-30.0
    image, mask = prepair_data.prepair('images/' + str(feature_index) + '_' + str(frame_index) + '.png',
                                       'images/' + str(feature_index) + '_' + str(frame_index) + '.txt', scale, alpha)
    data_input[i, 0, :, :] = image[:, :, 0]
    data_input[i, 1, :, :] = image[:, :, 1]
    data_input[i, 2, :, :] = image[:, :, 2]
    data_output[i, feature_index, :, :] = mask[:, :]

for iteration in range(100000):
    input=numpy.zeros((batch_size,3,config.detector_input_W,config.detector_input_W))
    output = numpy.zeros((batch_size,config.num_features, config.detector_output_W, config.detector_output_W))

    for j in range(batch_size):
        i=random.randint(0,N-1)
        input[j,:,:,:]=data_input[i,:,:,:]
        output[j, :, :, :] = data_output[i, :, :, :]


    optimizer.zero_grad()

    input_gpu = torch.from_numpy(input).float().cuda()
    target_gpu=torch.from_numpy(output).float().cuda()

    output_gpu=model.forward(input_gpu)

    loss=criterion(output_gpu,target_gpu)
    loss.backward()
    optimizer.step()


    if iteration%100==0:
        print(iteration, ' loss ',loss.item())
        out = output_gpu.data.cpu().numpy()
        image = out[0, 0]
        cv2.imshow('1', (image - image.min()) / (image.max() - image.min()))
        cv2.imshow('2', input[0, 0, :, :])
        cv2.imshow('3',output[0, 0, :, :])
        cv2.waitKey(100)

        torch.save(model.state_dict(),'detector.pth')


