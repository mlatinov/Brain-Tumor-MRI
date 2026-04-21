import tensorflow as tf 

def transfer_inceptionV3(input_shape) :

    # Get the base model without the head 
    inception_base = tf.keras.applications.InceptionV3(
        input_shape = input_shape,
        include_top = False,
        weights     = "imagenet"
    )

    # Freeze the Base 
    inception_base.trainable = False

    # Make the head 
    inputs  = tf.keras.Input(shape = input_shape)
    x  = inception_base(inputs, training = False)
    x  = tf.keras.layers.GlobalAveragePooling2D()(x)
    x  = tf.keras.layers.Dropout(rate = 0.4)(x)
    x  = tf.keras.layers.Dense(units = 64, activation = "relu")(x) 
    outputs = tf.keras.layers.Dense(units = 4, activation = "softmax")(x)
    
    # Ensamble the model 
    model = tf.keras.Model(inputs, outputs)

    return model

