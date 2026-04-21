import mlflow 
import tensorflow as tf 
from scripts.helpers.process_image import process_image
from scripts.cnns_models.ResNet50 import transfer_ResNet50

def exp_ResNet50() : 

    # Autolog the model specifications
    mlflow.keras.autolog()

    # Start the experiment
    with mlflow.start_run(run_name = "ResNet50 Baseline") :

        # Process the image data 
        model_data = process_image(
            training_dir = "data/Training/",
            testing_dir  = "data/Testing/",
            target_image_size = (224, 224),
            rescale           = None,
            color_mode        = "rgb"
        )

        # Get the model 
        model_resnet = transfer_ResNet50(input_shape = (224, 224, 3))

        ## Stage 1 Train the Head of the model 

        # Compile the model 
        model_resnet.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics = [
                tf.keras.metrics.AUC(multi_label = True, num_labels = 4),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

        # Train the head 
        history_stage_1 = model_resnet.fit(
            x               = model_data["training_set"],
            validation_data = model_data["validation_set"],
            epochs          = 15,
            verbose         = 1,
            callbacks       = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True,
                    verbose              = 1
                )
            ]
        )

        ## Stage 2 Fine Tune

        # Unfreeze top 30 layers 
        base = model_resnet.get_layer("resnet50")
        for layer in base.layers[-30 :] :
            layer.trainable = True

        # Compile the model 
        model_resnet.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics = [
                tf.keras.metrics.AUC(multi_label = True, num_labels = 4),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

        # Fine tune the model 
        history_stage_2 = model_resnet.fit(
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
                    monitor  = "val_loss",
                    factor   = 0.5,
                    patience = 2,
                    verbose  = 1
                )
            ]
        )