from keras.applications import DenseNet121
from keras import models
from keras import layers

def get_model():
    conv_base = DenseNet121(weights=None,
                            include_top=False,
                            input_shape=(224, 224, 3))

    x = layers.GlobalAveragePooling2D()(conv_base.output)
    x = layers.Dense(9, activation='softmax', name='fc_out')(x)

    model = models.Model(inputs=conv_base.input, outputs=x)

    return model