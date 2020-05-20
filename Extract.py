import torch.nn as nn
import torch.nn.functional as F
import torch

import re
import math
import collections
from functools import partial

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams',
                                      ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                                       'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
                                       'drop_connect_rate', 'depth_divisor', 'min_depth'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                                                 'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # mostly divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert p >= 0 and p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None:
        return None

    image_height, image_width = input_image_size if isinstance(input_image_size, tuple) else (
    input_image_size, input_image_size)
    #     print(f'input image height : {image_height}, input image_width : {image_width}')
    #     print(f'input stride : {stride}')
    height_stride = stride[0]
    width_stride = stride[1]

    image_height = int(math.ceil(image_height / height_stride))
    image_width = int(math.ceil(image_width / width_stride))
    return (image_height, image_width)


def get_same_padding_conv2d(image_size=None):
    return partial(Conv2dStaticSamePadding, image_size=image_size)


def efficientnet_params(model_name):
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),  # input 시점 resolution
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]

class EfficientNet(nn.Module):

    def __init__(self):
        super().__init__()



        self.model_name = 'efficientnet-b0'
        self.efficientnet_params = efficientnet_params(self.model_name)
        # efficientnet_params = efficientnet_params(model_name)

        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s21_e6_i16_o24_se0.25',
            'r2_k5_s11_e6_i24_o40_se0.25',
            'r3_k3_s21_e6_i40_o80_se0.25',
            'r3_k5_s22_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]

        global_params = GlobalParams(width_coefficient=self.efficientnet_params[0], depth_coefficient=self.efficientnet_params[1],
                                     image_size=self.efficientnet_params[2], drop_connect_rate=0.2, dropout_rate=0.2,
                                     depth_divisor=8,
                                     num_classes=91, batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, min_depth=None)

        self._global_params = global_params
#         self._blocks_args = blocks_args
        self._blocks_args = BlockDecoder.decode(blocks_args)

        # batch norm params
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # get stem static or dynamic convolution depending on image size
        image_size = self._global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # mostly 3,  differ from input_filters !
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, [2, 2])

        # build blocks
        self._blocks = nn.ModuleList([])
        stage = 2
        for block_args in self._blocks_args:
            #             print(f'on stage : {stage}')
            # update block input and output fitlers based on depth multiplier
            block_args = block_args._replace(input_filters=round_filters(block_args.input_filters, self._global_params),
                                             output_filters=round_filters(block_args.output_filters,
                                                                          self._global_params),
                                             num_repeat=round_repeats(block_args.num_repeat, self._global_params))

            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=[1, 1])
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            stage += 1

        # head
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d((1, 15))  
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs):

        bs = inputs.size(0)
        #         x = self.extract_feature(inputs)
        # stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # head
        x = self._swish(self._bn1(self._conv_head(x)))
   

        # pooling and final_linear layer
        x = self._avg_pooling(x)
        #         x = x.view(bs, -1)
        #         x = self._dropout(x)
        #         x = self._fc(x)

        return x

class MBConvBlock(nn.Module):

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # expansion phase (inverted bottle neck)
        input_filters = self._block_args.input_filters
        output_filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=input_filters, out_channels=output_filters, kernel_size=1,
                                       bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=output_filters, momentum=self._bn_mom, eps=self._bn_eps)

        # depth wise conv phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=output_filters, out_channels=output_filters, groups=output_filters,
                                      # groups makes it depthwise
                                      kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=output_filters, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # squeeze and excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=output_filters, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=output_filters, kernel_size=1)

        # pointwise conv phase (aka 1 x 1 conv)
        final_output = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=output_filters, out_channels=final_output, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_output, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # drop_connect_rate  : float, between 0 and 1

        # expansion and depthwise convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # squeeze and excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # point wise colv
        x = self._project_conv(x)
        x = self._bn2(x)

        # skip connection and drop connect
        input_filter, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filter == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection

        return x

    def set_swish(self, memory_efficient=True):
        # sets swish function as memory efficient (for training) or standard (for export)

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()



class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        return blocks_args

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(x) for x in list(options['s'])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))


class Conv2dStaticSamePadding(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride

        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))




    
    
############################################################################
#---------------------------RCNN CODE BELOW--------------------------------#
############################################################################

class GRCL(nn.Module):
    
    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias = False)
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias = False)
        
        self.BN_x_init = nn.BatchNorm2d(output_channel)
        self.num_iteration = num_iteration
        
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)
        
    def forward(self, input):
        
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.BN_x_init(wf_u))
        
        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))
            
        return x
    
class GRCL_unit(nn.Module):
    
    def __init__(self, output_channel):
        
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)
        
    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = F.sigmoid(G_first_term + G_second_term)
        
        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = F.relu(x_first_term + x_second_term)
        
        return x
    
    
class RCNN_extractor(nn.Module):
    
    def __init__(self, input_channel, output_channel = 512):
        super(RCNN_extractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4), 
                               int(output_channel / 2), output_channel] #[64, 128, 256, 512]
        
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2,2), # 64 * 16 * 50
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2,2), #64 * 8 * 25
            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2,1), (0,1)), #128* 4 * 26
            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2,1), (0,1)), #256* 2 * 27
            
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True) # 512 * 1 * 26
            )
        
    def forward(self, input):
        return self.ConvNet(input)
 


            