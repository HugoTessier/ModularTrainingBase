from semantic_segmentation_utils.cityscapes_dataset import load_cityscapes


def load(path, train_batch_size, test_batch_size):
    return load_cityscapes(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        train_crop_size=(512, 1024),
        test_crop_size=(1024, 2048),
        dataset_root=path,
        workers=4,
        train_label_downsample_rate=1,
        train_image_downsample_rate=1,
        test_label_downsample_rate=1,
        test_image_downsample_rate=1)


OBJECT = load
