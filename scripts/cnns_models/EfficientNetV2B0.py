
import tensorflow as tf

def transfer_efficient_net(input_shape) : 

    # Load the Pretrained model 
    efficient_net = tf.keras.applications.EfficientNetV2B0(
        include_top = False,
        weights     = "imagenet",
        input_shape = input_shape
    )

    # Freeze the Base
    efficient_net.trainable = False

    # Build the Head 
    input = tf.keras.Input(shape = input_shape)
    x = efficient_net(input, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate = 0.4)(x)
    x = tf.keras.layers.Dense(units = 64, activation = "relu")(x)
    output = tf.keras.layers.Dense(units = 4, activation = "softmax")(x)

    # Build the model 
    model = tf.keras.Model(input, output)

    return model 
