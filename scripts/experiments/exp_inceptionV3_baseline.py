import mlflow
import tensorflow as tf
from scripts.helpers.process_image import process_image
from scripts.cnns_models.InceptionV3 import transfer_inceptionV3

def exp_inceptionV3() :

    # Auto Log model specification 
    mlflow.keras.autolog()

    # Start the Experiment
    with mlflow.start_run(run_name = "Inception V3 Baseline") :

        # Process image data 
        model_data = process_image(
            training_dir = "data/Training/",
            testing_dir  = "data/Testing/",
            target_image_size = (224, 224),
            rescale           =  None,
            color_mode        = "rgb"
        )

        # Get the model 
        inception_model = transfer_inceptionV3(input_shape = (224, 224, 3))

        ## Stage 1 Train the Head of the model
         
        # Compile the model 
        inception_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics   = [
                tf.keras.metrics.AUC(multi_label = True, num_labels = 4),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]  
        )
        # Train the Head 
        history_stage_1 = inception_model.fit(
            x                = model_data["training_set"],
            validation_data  = model_data["validation_set"],
            epochs           = 15,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True,
                    verbose              = 1
                )
            ]
        )

        ## Stage 2 Fine tuen the model 

        # Unfreeze the last 62 layers 
        base = inception_model.get_layer("inception_v3")  
        base.trainable = True
        
        for layer in base.layers[:249]: 
            layer.trainable = False

        # Compile the model again 
        inception_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics   = [
                tf.keras.metrics.AUC(multi_label = True, num_labels = 4),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ] 
        )

        # Fine tune the model 
        history_stage_2 = inception_model.fit(            
            x               = model_data["training_set"],
            validation_data = model_data["validation_set"],
            epochs          = 15,
            verbose         = 1,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True,
                    verbose              = 1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor   = "val_loss",
                    factor    = 0.5,
                    patience  = 3,
                    verbose   = 1
                )
            ]
        )
