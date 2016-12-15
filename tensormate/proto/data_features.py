from tensormate.proto import feature


class ImageProperties(feature.Features):
    image_format = feature.BytesFeature(replace={"_": "/"})
    image_filename = feature.BytesFeature(replace={"_": "/"})
    image_height = feature.Int64Feature(replace={"_": "/"})
    image_width = feature.Int64Feature(replace={"_": "/"})
    image_channels = feature.Int64Feature(replace={"_": "/"}, default=3)


class EncodedImage(feature.Features):
    image_encoded = feature.BytesFeature(replace={"_": "/"})


class ImageClass(feature.Features):
    image_class_label = feature.Int64Feature(replace={"_": "/"})
    image_class_text = feature.BytesFeature(replace={"_": "/"})


class BoundingBoxes(feature.Features):
    image_object_bbox_xmin = feature.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_xmax = feature.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_ymin = feature.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_ymax = feature.SparseFloat32Feature(replace={"_": "/"})
    image_object_bbox_label = feature.SparseInt64Feature(replace={"_": "/"})
    # image_object_bbox_encode = feature.BytesFeature(replace={"_": "/"})


class EncodedImageFeatures(EncodedImage, ImageProperties):
    pass


class LabelFeatures(feature.Features):
    label_index = feature.Int64Feature(replace={"_": "/"})