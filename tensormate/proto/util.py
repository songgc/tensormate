from tensorflow.core.framework import summary_pb2


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def make_encoded_image_summary(name, encoded_image_string, height, width, colorspace=3):
    image = summary_pb2.Summary.Image(height=height,
                                      width=width,
                                      colorspace=colorspace,
                                      encoded_image_string=encoded_image_string)
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, image=image)])
