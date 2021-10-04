import tensorflow as tf
import os
from tensorflow_examples.models.pix2pix import pix2pix

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def generate_unet(img_size):
    output_channels = 4      # number of classes on the dataset
    base_model = tf.keras.applications.MobileNetV2(input_shape=[img_size, img_size, 3], 
                                                   include_top=False)
    layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
         output_channels, 3, strides=2,
         padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generate_unet_512(img_size=512):    
    output_channels = 4      # number of classes on the dataset
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], 
                                                   include_top=False)
    layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])
    mid = tf.keras.layers.MaxPooling2D(3, (4,4))(inputs)
    x = tf.keras.layers.Conv2D(3, (3,3), padding = 'same')(mid)

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')(x)  #64x64 -> 128x128
    mid = tf.keras.layers.UpSampling2D((4,4))(last)
    mid_x = tf.keras.layers.Conv2D(4, (12,12), padding= 'same')(mid)
    x = tf.keras.layers.Conv2D(4, (3,3), padding= 'same')(mid_x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)