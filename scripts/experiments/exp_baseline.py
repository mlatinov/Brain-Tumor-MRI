
import mlflow
import tensorflow as tf
from scripts.helpers.process_image import process_image
from scripts.helpers.plot_image_aug import plot_augmented_samples
from scripts.cnns_models.EfficientNetV2B0 import transfer_efficient_net

def exp_baseline() :

    # Autolog Model Details
    mlflow.keras.autolog()

    # Run the Experiment 
    with mlflow.start_run(run_name = "Efficient Net Baseline") : 

        # Process the images 
        model_data = process_image(
            training_dir = "data/Training/",
            testing_dir  = "data/Testing/",
            rescale      =  None, # Efficient Net handles it internally 
            color_mode   = "rgb", # To match the model chanels expectations 
            target_image_size = (224, 224)
        )

        # Plot and save augmented image 
        image_augmented = plot_augmented_samples(
            data_generator = model_data["training_set"],
            n_samples      = 4
        )

        # Get the model 
        effecient_net = transfer_efficient_net(input_shape = (224, 224, 3))

       # Compile the model for Stage one Training the Head 
        effecient_net.compile(
            optimizer =  tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics   = [
                tf.keras.metrics.AUC(multi_label = True, num_labels = 4),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision()
            ]
        )
        ## Stage 1 Train the head 
        history_stage_1 = effecient_net.fit(
            x                = model_data["training_set"],
            validation_data  = model_data["validation_set"],
            epochs           = 10,
            verbose          = 1,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True,
                    verbose              = 1
                )
            ]
        )
        ## Stage 2 Fine Tune 

        # Unfreeze the last 20 layers 
        base = effecient_net.get_layer("efficientnetv2-b0")
        
        # Unfreeze last 20 layers
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False

        # Recompile the model with Lower LR
        effecient_net.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics   = [
                tf.keras.metrics.AUC(multi_label = True, num_labels = 4),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

        # Retrain the model with unfreezed layers
        history_stage2 = effecient_net.fit(
             x               = model_data["training_set"],
             validation_data = model_data["validation_set"],
             epochs          = 10,
             callbacks       = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor  = "val_loss",
                    factor   = 0.5,
                    patience = 2
                )
            ]
        )