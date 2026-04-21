import tensorflow as tf

def transfer_ResNet50(input_shape) :

    # Get the model without the head 
    resNet50 = tf.keras.applications.ResNet50(
        include_top = False,
        weights     = "imagenet",
        input_shape = input_shape
    )

    # Freeze the Base 
    resNet50.trainable = False

    # Make a new head 
    inputs = tf.keras.Input(shape = input_shape)
    x = resNet50(inputs, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate = 0.4)(x)
    x = tf.keras.layers.Dense(units = 64, activation = "relu")(x)
    output = tf.keras.layers.Dense(units = 4, activation = "softmax")(x)

    # Ensamble the model 
    model = tf.keras.Model(inputs, output)

    return model 

