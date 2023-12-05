import tensorflow as tf

class Yolo:

    def __init__(self, n_classes, model_size, max_output_size, data_format):

        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.data_format = data_format