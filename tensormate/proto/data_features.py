from tensormate.proto import features


class ImageProperties(features.Features):
    image_format = features.BytesFeature(replace={"_": "/"})
    image_filename = features.BytesFeature(replace={"_": "/"})
    image_height = features.Int64Feature(replace={"_": "/"})
    image_width = features.Int64Feature(replace={"_": "/"})
    image_channels = features.Int64Feature(replace={"_": "/"}, default=3)


class EncodedImage(features.Features):
    image_encoded = features.BytesFeature(replace={"_": "/"})


class ImageClass(features.Features):
    image_class_label = features.Int64Feature(replace={"_": "/"})
    image_class_text = features.BytesFeature(replace={"_": "/"})


class BoundingBoxes(features.Features):
    image_object_bbox_xmin = features.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_xmax = features.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_ymin = features.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_ymax = features.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_label = features.SparseInt64Feature(replace={"_": "/"})
    # image_object_bbox_encode = feature.BytesFeature(replace={"_": "/"})


class EncodedImageFeatures(EncodedImage, ImageProperties):
    pass


class LabelFeatures(features.Features):
    label_index = features.Int64Feature(replace={"_": "/"})