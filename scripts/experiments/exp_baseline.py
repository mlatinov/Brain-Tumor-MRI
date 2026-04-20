import mlflow
import tensorflow as tf
from scripts.helpers.process_image import process_image
from scripts.helpers.plot_image_aug import plot_augmented_samples

def exp_baseline() :

    # Autolog Model Details
    mlflow.keras.autolog()

    # Run the Experiment 
    with mlflow.start_run(run_name = "Baseline") : 

        # Process the images 
        model_data = process_image(
            training_dir = "data/Training/",
            testing_dir  = "data/Testing/",
            target_image_size = (254, 254)
        )

        # Plot and save augmented image 
        image_augmented = plot_augmented_samples(
            data_generator = model_data["training_set"],
            n_samples      = 4
        )