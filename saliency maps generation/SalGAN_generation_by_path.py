import torch
import torch.nn as nn
import numpy as np
import cv2
import os

class SalGan(nn.Module):
    def __init__(self):
        super(SalGan, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.max_pool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.sigmoid = nn.Sigmoid()

        # conv_1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)
        # conv_2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)

        # conv_3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding = 1)

        # conv_4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding = 1)

        # conv_5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding = 1)

        # deconv_5
        self.uconv5_3 = nn.Conv2d(512, 512, 3, padding = 1)
        self.uconv5_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.uconv5_1 = nn.Conv2d(512, 512, 3, padding = 1)

        # deconv_4
        self.uconv4_3 = nn.Conv2d(512, 512, 3, padding = 1)
        self.uconv4_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.uconv4_1 = nn.Conv2d(512, 512, 3, padding = 1)

        # deconv_3
        self.uconv3_3 = nn.Conv2d(512, 256, 3, padding = 1)
        self.uconv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.uconv3_1 = nn.Conv2d(256, 256, 3, padding = 1)

        # deconv_2
        self.uconv2_2 = nn.Conv2d(256, 128, 3, padding = 1)
        self.uconv2_1 = nn.Conv2d(128, 128, 3, padding = 1)

        # deconv_1
        self.uconv1_2 = nn.Conv2d(128, 64, 3, padding = 1)
        self.uconv1_1 = nn.Conv2d(64, 64, 3, padding = 1)

        # output
        self.output = nn.Conv2d(64, 1, 1, padding = 0)

    def forward(self, x):
        #assert(x.size()[1:] == (3, 192, 256))
        #print 'input', x.size()
        # todo insert data pre-processing: subtract image net mean image

        # conv_1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pool2d(x)
        #print 'pool1: ', x.size()

        # conv_2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pool2d(x)
        #print 'pool2: ', x.size()

        # conv_3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.max_pool2d(x)
        #print 'pool3: ', x.size()

        # conv_4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.max_pool2d(x)
        #print 'pool4: ', x.size()

        # conv_5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        #print 'conv5: ', x.size()

        # deconv_5
        x = self.relu(self.uconv5_3(x))
        x = self.relu(self.uconv5_2(x))
        x = self.relu(self.uconv5_1(x))
        #print 'uconv5: ', x.size()

        # pool 4
        x = self.up_sample(x)
        #print 'upool4: ', x.size()

        # deconv 4
        x = self.relu(self.uconv4_3(x))
        x = self.relu(self.uconv4_2(x))
        x = self.relu(self.uconv4_1(x))
        #print 'uconv4: ', x.size()

        # pool 3
        x = self.up_sample(x)
        #print 'upool3: ', x.size()

        # deconv 3
        x = self.relu(self.uconv3_3(x))
        x = self.relu(self.uconv3_2(x))
        x = self.relu(self.uconv3_1(x))
        #print 'uconv3: ', x.size()

        # pool 2
        x = self.up_sample(x)
        #print 'upool2: ', x.size()

        # deconv 2
        x = self.relu(self.uconv2_2(x))
        x = self.relu(self.uconv2_1(x))
        #print 'uconv2: ', x.size()

        # pool 1
        x = self.up_sample(x)
        #print 'upool1: ', x.size()

        # deconv 1
        x = self.relu(self.uconv1_2(x))
        x = self.relu(self.uconv1_1(x))
        #print 'uconv1: ', x.size()

        # output
        x = self.sigmoid(self.output(x))
        #print 'output: ', x.size()

        return x

def theano_conv_2_torch_tensor(torch_dict, weights_dict, w_name, b_name, conv_name, flip_filter = False):
    weight = weights_dict[w_name]
    bias = weights_dict[b_name]
    if flip_filter:
        weight = weight[:, :, ::-1, ::-1].copy() # important
        print(weight.shape)
        weight = torch.from_numpy(weight)
    else:
        weight = torch.from_numpy(weight)

    bias = torch.from_numpy(bias)
    print(weight.size(), bias.size())
    print(torch_dict[conv_name + '.weight'].size(), torch_dict[conv_name+'.bias'].size())
    assert torch_dict[conv_name + '.weight'].size() == weight.size()
    assert torch_dict[conv_name + '.bias'].size() == bias.size()
    torch_dict[conv_name + '.weight'] = weight
    torch_dict[conv_name + '.bias'] = bias

def load_npz_weights(torch_dict, weights):
    conv_name = ['conv1_1', 'conv1_2',
                 'conv2_1', 'conv2_2',
                 'conv3_1', 'conv3_2', 'conv3_3',
                 'conv4_1', 'conv4_2', 'conv4_3',
                 'conv5_1', 'conv5_2', 'conv5_3']
    uconv_name =['u' + name for name in conv_name[::-1]]

    # convert encoder
    for i, name in zip(range(len(conv_name)), conv_name):
        print('arr_%d, arr_%d, %s'%(2 * i, 2 * i + 1, name))
        theano_conv_2_torch_tensor(torch_dict, weights,
                                   'arr_%d' % (2 * i), 'arr_%d' % (2 * i + 1), name, False)

    # convert decoder
    offset = len(conv_name) * 2
    for i, name in zip(range(len(uconv_name)), uconv_name):
        print('arr_%d, arr_%d, %s' % (2 * i + offset, 2 * i + 1 + offset, name))
        theano_conv_2_torch_tensor(torch_dict, weights,
                                   'arr_%d' % (2 * i + offset),'arr_%d' % (2 * i + 1 + offset), name, True)

    # convert output
    theano_conv_2_torch_tensor(torch_dict, weights, 'arr_52', 'arr_53', 'output', True)

def image_preprocess(image, resized_height = 192, resized_width = 256):
    """ 
    Args: image is BGR format, [h, w, 3]
    
    return a mean-subtract tensor, [3, h, w]
    """
    bgr_mean = np.array([103.939, 116.779, 123.68]).astype(np.float32)
    image = cv2.resize(image, (resized_width, resized_height), interpolation = cv2.INTER_AREA)

    image = np.transpose(image, [2, 0, 1])
    image = image.astype(np.float32)
    image -= bgr_mean[:, np.newaxis, np.newaxis]
    X = torch.from_numpy(image)
    return X

def post_process(saliency_map, orig_height, orig_width):
    """ 
    Args: saliency_map is numpy 2-d array with shape [h, w] 0 ~ 1
    
    return cv2 gray image saliency_map
    """
    saliency_map = (saliency_map * 255).astype(np.uint8)

    # resize back to original size
    saliency_map = cv2.resize(saliency_map, (orig_width, orig_height), interpolation = cv2.INTER_CUBIC)
    # blur
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    return saliency_map

def SaliencyNet(pretrained = False, **kwargs):
    model = SalGan()
    if pretrained:
        model.load_state_dict(torch.load('gan_torch_model.pkl'))
    return model

# 1. Define the root_dir and result_dir
# 2. Run
# 3. The processed image will be saved to result_dir one by one
# 4. The corresponding saliency maps have the same names of RGB images
if __name__ == '__main__':

    gan = SaliencyNet(pretrained = True)

    # This is an example, you should change it with your own path
    root_dir = 'D:\\dataset\\ILSVRC2012_train_256\\'
    result_dir = 'D:\\dataset\\ILSVRC2012_train_256_saliencymaps\\'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    images = os.listdir(root_dir)
    
    for image_path in images:
        image = cv2.imread(os.path.join(root_dir, image_path), cv2.IMREAD_COLOR)
        size = image.shape[:2]

        X = image_preprocess(image)
        X = torch.autograd.Variable(X.unsqueeze(0))

        result = gan(X)
    
        saliency_map = post_process(result.data.numpy()[0, 0], size[0], size[1])
        cv2.imwrite(os.path.join(result_dir, image_path), saliency_map)
