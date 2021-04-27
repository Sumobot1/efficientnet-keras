import collections

# Found on the internet - it looks like Variance Scaling
# Read: https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

# Parameters for each block of the EfficientNet-B0 baseline network (Table 1 in paper)
BlockParams = collections.namedtuple('BlockParams', [
    'depthwise_kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'depthwise_strides', 'se_ratio'
])
BASELINE_NETWORK_PARAMS = [
    BlockParams(depthwise_kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, depthwise_strides=[1, 1], se_ratio=0.25),
    BlockParams(depthwise_kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, depthwise_strides=[2, 2], se_ratio=0.25),
    BlockParams(depthwise_kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, depthwise_strides=[2, 2], se_ratio=0.25),
    BlockParams(depthwise_kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, depthwise_strides=[2, 2], se_ratio=0.25),
    BlockParams(depthwise_kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, depthwise_strides=[1, 1], se_ratio=0.25),
    BlockParams(depthwise_kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, depthwise_strides=[2, 2], se_ratio=0.25),
    BlockParams(depthwise_kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, depthwise_strides=[1, 1], se_ratio=0.25)
]