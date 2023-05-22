from necessary_for_cityscapes.models.utils.resnet_unet import resnet18_unet

OBJECT = resnet18_unet(num_classes=19,
                      pretrained_encoder=True,
                      output_downsampling_rate=4,
                      entry_downsampling_rate=1)