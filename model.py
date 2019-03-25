from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Input, Concatenate
from keras.models import Model

from keras_contrib.layers.normalization import InstanceNormalization

import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0],
                s[2] + 2 * self.padding[1], s[3])
    
    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad],
                          [w_pad, w_pad], [0, 0]], 'REFLECT')


def get_gen_model():
    inp_im = Input(shape=(1200, 1600, 3), name='Input_image')

    inp_enh256 = Input(shape=(1200, 1600, 3), name='enh256')
    inp_enh128 = Input(shape=(1200, 1600, 3), name='enh128')

    # compute A
    # enc
    xA = Conv2D(8, (8, 8), padding='valid', strides=(4, 4))(inp_im)
    xA = LeakyReLU(alpha=0.15)(xA)
    xA = Conv2D(16, (8, 8), padding='valid', strides=(4, 4))(xA)
    xA = InstanceNormalization(axis=-1)(xA)
    xA = LeakyReLU(alpha=0.15)(xA)

    # dec
    xA = Conv2DTranspose(16, (8, 8), padding='valid', strides=(4, 4))(xA)
    xA = InstanceNormalization(axis=-1)(xA)
    xA = LeakyReLU(alpha=0.15)(xA)
    xA = Conv2DTranspose(16, (4, 4), padding='valid', strides=(1, 1))(xA)
    xA = InstanceNormalization(axis=-1)(xA)
    xA = LeakyReLU(alpha=0.15)(xA)
    xA = Conv2DTranspose(8, (8, 8), padding='valid', strides=(4, 4))(xA)
    xA = InstanceNormalization(axis=-1)(xA)
    xA_out = LeakyReLU(alpha=0.15)(xA)

    # t path
    xt = ReflectionPadding2D(padding=(1, 1))(inp_im)
    xt = Conv2D(16, (3, 3), padding='valid')(xt)
    xt = LeakyReLU(alpha=0.15)(xt)

    xt = Concatenate()([xt, xA_out])

    xt = ReflectionPadding2D(padding=(1, 1))(xt)
    xt = Conv2D(16, (3, 3), padding='valid')(xt)
    xt = InstanceNormalization(axis=-1)(xt)
    xt = LeakyReLU(alpha=0.15)(xt)
    xt = ReflectionPadding2D(padding=(1, 1))(xt)
    xt = Conv2D(16, (3, 3), padding='valid')(xt)
    xt = InstanceNormalization(axis=-1)(xt)
    xt = LeakyReLU(alpha=0.15)(xt)
    xt_out = xt

    # enh ims
    xe2 = ReflectionPadding2D(padding=(1, 1))(inp_enh256)
    xe2 = Conv2D(16, (3, 3), padding='valid')(xe2)
    xe2 = LeakyReLU(alpha=0.15)(xe2)
    xe2 = ReflectionPadding2D(padding=(1, 1))(xe2)
    xe2 = Conv2D(16, (3, 3), padding='valid')(xe2)
    xe2 = InstanceNormalization(axis=-1)(xe2)
    xe2 = LeakyReLU(alpha=0.15)(xe2)
    
    xe1 = ReflectionPadding2D(padding=(1, 1))(inp_enh128)
    xe1 = Conv2D(16, (3, 3), padding='valid')(xe1)
    xe1 = LeakyReLU(alpha=0.15)(xe1)
    xe1 = ReflectionPadding2D(padding=(1, 1))(xe1)
    xe1 = Conv2D(16, (3, 3), padding='valid')(xe1)
    xe1 = InstanceNormalization(axis=-1)(xe1)
    xe1 = LeakyReLU(alpha=0.15)(xe1)

    # all paths merge here
    merged_x = Concatenate()([xt_out, xA_out, xe1, xe2])
    merged_x = ReflectionPadding2D(padding=(1, 1))(merged_x)
    merged_x = Conv2D(64, (3, 3), padding='valid')(merged_x)
    merged_x = InstanceNormalization(axis=-1)(merged_x)
    merged_x = LeakyReLU(alpha=0.15)(merged_x)
    merged_x = ReflectionPadding2D(padding=(1, 1))(merged_x)
    merged_x = Conv2D(32, (3, 3), padding='valid')(merged_x)
    merged_x = InstanceNormalization(axis=-1)(merged_x)
    merged_x = LeakyReLU(alpha=0.15)(merged_x)
    merged_x = ReflectionPadding2D(padding=(1, 1))(merged_x)
    merged_x = Conv2D(3, (3, 3), padding='valid',
                      activation='sigmoid')(merged_x)

    model = Model(inputs=[inp_im, inp_enh128, inp_enh256], outputs=[merged_x])

    return model


if __name__ == '__main__':
    model = get_gen_model()
    model.summary()

    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model.png')
    
