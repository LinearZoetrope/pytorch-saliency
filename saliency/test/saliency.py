import torch
from saliency.saliency import Saliency
import numpy as np
from scipy.ndimage import label
import torchvision


class TestSaliency(Saliency):

    def __init__(self, model):
        super(TestSaliency, self).__init__(model)


    def generate_saliency(self, input, target):

        #self.model.zero_grad()

        output = self.model(input)
        #print(output)
        #grad_outputs = torch.zeros_like(output)

        #grad_outputs[:, target] = 1
        #print('grad outputs')
        #print(grad_outputs)

        #print(input)
        #torch.set_printoptions(threshold=10000)
        #print(input.numpy().shape)
        #f = open('input.txt', 'w+')
        #f.write(str(input.numpy()))
        #f.close()

        #index: 0 layer: HP
        # index: 1 layer: Ship
        # index: 2 layer: Small Towers
        # index: 3 layer: Big Towers
        # index: 4 layer: Small Cities
        # index: 5 layer: Big Cities
        # index: 6 layer: Friend
        # index: 7 layer: Enemy


        #return (input.grad.clone()[0] * input)
        input2 = input.clone()
        image = np.zeros((40, 40))
        input2 = input2.view(40, 40, 8)

        #logical or of the input images to get the original image
        for i in range(8):
            if i!= 0:
                image = np.logical_or(image, input2[:, :, i].numpy())*1

        #get the number of objects in the image
        labeled_array, num_objects = label(image)

        indices = []
        for i in range(num_objects):
            indices.append(np.argwhere(labeled_array == i+1))

        #print('object 1\n')
        #print(indices[0])
        #print('object 2\n')
        #print(indices[1])
        #print('object 3\n')
        #print(indices[2])
        #print('object 4\n')
        #print(indices[3])
        #print('object 5\n')
        #print(indices[4])




        #OR the images to get the location of all
        #input2 = input2.numpy()
        f = open('labeled_array.txt', 'w')
        f.write('\n\n\n')

        for i in range(labeled_array.shape[0]):
            for j in range(labeled_array.shape[1]):
                f.write(str(labeled_array[i,j]))
            f.write('\n')
        f.close()

        saliencies = torch.zeros_like(input)
        saliencies = saliencies.view(40, 40, 8)
        input2 = input.clone()
        input2 = input2.view(40, 40, 8)

        #index: 0 layer: HP
        #index: 1 layer: Agent Location
        #index: 2 layer: Small Towers
        #index: 3 layer: Big Towers
        #index: 4 layer: Friend
        #index: 5 layer: Enemy


        for i in range(8):
            for j in range(num_objects):
                for k in range(indices[j].shape[0]):
                    x = indices[j][k][0]
                    y = indices[j][k][1]
                    # print('x: '+str(x)+' y: '+str(y))
                    # print('Value of input: '+str(input2[:, :, i][x][y]))
                    if i!=0:
                        if input2[:, :, i][x][y] == 1:
                            input2[:, :, i][x][y] = 0
                        else:
                            input2[:, :, i][x][y] = 1

                    else: #binary variables
                        temp = 0.3*input2[:, :, i][x][y]
                        input2[:, :, i][x][y] += temp
                perturbed_output = self.model(input2.view(1, 12800))
                saliency = (perturbed_output - output)
                if i==0:
                    saliency = saliency/temp
                #print(saliency)
                input2 = input.clone()
                input2 = input2.view(40, 40, 8)

                for k in range(indices[j].shape[0]):
                    x = indices[j][k][0]
                    y = indices[j][k][1]
                    saliencies[:, :, i][x][y] = saliency[:, target]
                #print(saliency[0][target])



                    #for l in range(6):
                    #    torchvision.utils.save_image(saliency[:, :, l], "Image perturbed: "+str(i) + "/" + "object perturbed: "+str(j)+ "/" + str(l) + ".png", normalize=True)




        return (saliencies.view(1, 12800))
