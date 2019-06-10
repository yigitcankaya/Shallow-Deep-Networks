# gradcam_experiments.py
# runs the experiments in section 5.2 
# visualizes the disagreement and confusing input elements using GradCam

import cv2
import os

import torch
from torchvision.utils import save_image 
import numpy as np
import torch.nn as nn

from shutil import copyfile
from torch.autograd import Variable

import aux_funcs as af
import network_architectures as arcs

def save_gradient_images(gradient, path_to_file):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = gradient.transpose(1, 2, 0) * 255

    cv2.imwrite(path_to_file, np.uint8(gradient))

def preprocess_image(img):
    means = [0.4802,  0.4481,  0.3975]
    stds = [0.2302, 0.2265, 0.2262]
    
    preprocessed_img = img.copy()[: , :, ::-1]

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)

    return input

def save_gradcam(img, mask, path_to_file):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path_to_file, np.uint8(255 * cam))

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for idx, module in enumerate(self.model.layers):
            for idx_in, module_in in enumerate(module.layers):
                x = module_in(x)
                #print('{},{}'.format(idx, idx_in))
                #print(module_in)
                layer_name = '{},{}'.format(idx, idx_in)
                if '{},{}'.format(idx, idx_in) in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        x = self.model.init_conv(x)
        target_activations, output  = self.feature_extractor(x)
        output = self.model.end_layers(output)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            print('Predicted class: {}'.format(af.get_tinyimagenet_classes(index)))
        
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis = (2, 3))[0, :]


        cam = np.zeros(target.shape[1 : ], dtype = np.float32) + 1e-5

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (64, 64), interpolation=cv2.INTER_CUBIC)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(torch.nn.Module):

    def __init__(self):
        super(GuidedBackpropReLU, self).__init__()

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.input = input
        self.output = output
        return output

    def backward(self, grad_output):
        grad_input = None

        size = self.input.size()

        positive_mask_1 = (self.input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        
        grad_input = torch.addcmul(torch.zeros(size).type_as(self.input), torch.addcmul(torch.zeros(size).type_as(self.input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        # replace ReLU with GuidedBackpropReLU
        relu_locations = []
        for idx, module in enumerate(self.model.layers):
            for idx_in, module in enumerate(module.layers):
                if module.__class__.__name__ == 'ReLU':
                    relu_locations.append((idx, idx_in))
        
        for idx, idx_in in relu_locations:
            self.model.layers[idx].layers[idx_in] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        one_hot = torch.sum(one_hot * output)
        
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]
        return output


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    cam_mask = np.zeros(guided_backprop_mask.shape)
    for i in range(0, guided_backprop_mask.shape[0]):
        cam_mask[i, :, :] = grad_cam_mask

    cam_gb = np.multiply(cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    images_path = 'only_first'
    output_file_path = 'gradcam_output'
    af.create_path(output_file_path)

    # load the model
    target_layers = ['12,0']
    output_id = -1; save_name = 'final'

    models_path = 'networks/{}'.format(af.get_random_seed())
    sdn_name = 'tinyimagenet_vgg16bn_sdn_ic_only'
    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.eval()

    # uncomment this line to visualize the first internal classifier
    #output_id = 0; save_name = 'first'; sdn_model, sdn_params = af.sdn_prune(models_path, sdn_name, prune_after_output=output_id); target_layers = ['1,0']
    
    converted_cnn, converted_cnn_params = af.sdn_to_cnn(None, None, epoch=-1, preloaded=(sdn_model, sdn_params))

    for file_id, file_name in enumerate(sorted(os.listdir(images_path))):
        input_img_path = images_path+'/'+file_name
        print('Image: {}'.format(input_img_path))

        img = cv2.imread(input_img_path, 1)
        img = np.float32(img) / 255
        input = preprocess_image(img)
        
        # output file paths
        cur_output_file_path = '{}/{}'.format(output_file_path, file_id)
        orig_file_path = '{}/orig.jpg'.format(cur_output_file_path)
        af.create_path(cur_output_file_path)
        copyfile(input_img_path, orig_file_path)

        outputs = sdn_model(input)
        pred_indices = [np.argmax(softmax.cpu().data.numpy()) for softmax in outputs]
        pred_classes = [af.get_tinyimagenet_classes(index) for index in pred_indices]
        print(pred_classes)

        # output file names
        gcam_file_name = 'gcam_{}'.format(save_name)
        gb_file_name = 'gb_{}'.format(save_name)
        ggcam_file_name = 'ggcam_{}'.format(save_name)
        
        # apply gradcam and save it
        grad_cam = GradCam(model=converted_cnn, target_layer_names=target_layers)
        target_index = int(pred_indices[output_id]) # none means the predicted class will be visualized
        gcam_mask = grad_cam(input, target_index)
        path_to_file = os.path.join(cur_output_file_path, gcam_file_name+'.jpg')
        save_gradcam(img, gcam_mask, path_to_file)

        # apply guided backpropagation
        gb_model = GuidedBackpropReLUModel(model=converted_cnn)
        gb_mask = gb_model(input, index=target_index)
        path_to_file = os.path.join(cur_output_file_path, gb_file_name+'.jpg')
        save_gradient_images(gb_mask, path_to_file)

        # apply guided gradcam and save it
        ggcam_mask = guided_grad_cam(gcam_mask, gb_mask)
        path_to_file = os.path.join(cur_output_file_path, ggcam_file_name+'.jpg')
        save_gradient_images(ggcam_mask, path_to_file)