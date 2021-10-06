

def make_model(network, input_shape, output_channels):
    if network == 'mobileNetV2_Unet':
        from models.models import mobileNetV2_Unet
        return mobileNetV2_Unet(input_shape, output_channels)
    elif network == 'mobilenet_fcn_32':
        from models.models import mobilenet_fcn_32
        return mobilenet_fcn_32(input_shape, output_channels) 
    elif network == 'mobilenet_fcn_8':
        from models.models import mobilenet_fcn_8
        return mobilenet_fcn_8(input_shape, output_channels) 
    elif network == 'mobileNetV2_fcn_8':
        from models.models import mobileNetV2_fcn_8
        return mobileNetV2_fcn_8(input_shape, output_channels) 
    elif network == 'mobilenet_pspnet':
        from models.models import mobilenet_pspnet
        return mobilenet_pspnet(input_shape, output_channels) 
    elif network == 'mobileNetV2_pspnet':
        from models.models import mobileNetV2_pspnet
        return mobileNetV2_pspnet(input_shape, output_channels)
    elif network == 'resnet_pspnet':
        from models.PSPNet import resnet_pspnet
        return resnet_pspnet(input_shape, output_channels) 
    elif network == 'mobilenet_segnet':
        from models.models import mobilenet_segnet
        return mobilenet_segnet(input_shape, output_channels)
    elif network == 'mobileNetV2_segnet':
        from models.models import mobileNetV2_segnet
        return mobileNetV2_segnet(input_shape, output_channels) 
    elif network == 'Deeplabv3':
        from models.models import Deeplabv3
        return Deeplabv3(input_shape, output_channels) 
    else:
        raise ValueError('unknown network ' + network)