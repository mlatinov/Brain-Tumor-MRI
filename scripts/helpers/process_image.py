from tensorflow.keras.preprocessing.image import ImageDataGenerator

def process_image(training_dir, testing_dir, target_image_size, color_mode = "grayscale", rescale = 1./255) :
    
    # Create Image Generators 
    train_generator = ImageDataGenerator(
        width_shift_range    = 0.05,       
        height_shift_range   = 0.05,
        validation_split     = 0.2,
        brightness_range     = [0.8, 1.2],
        zoom_range           = 0.1,
        horizontal_flip      = True,
        rotation_range       = 10,
        rescale              = rescale
    )
    test_generator = ImageDataGenerator(rescale = rescale)

    # Apply them to the supplyed dir 
    train_data = train_generator.flow_from_directory(
        directory   = training_dir,
        color_mode  = color_mode,
        class_mode  = "categorical",
        target_size = target_image_size,
        batch_size  = 32,
        subset      ="training",
        seed        = 42,
    )
    validation_data = train_generator.flow_from_directory(
        directory      = training_dir,
        color_mode     = color_mode,
        class_mode     = "categorical",
        target_size    = target_image_size,
        batch_size     = 32,
        subset         = "validation",
        seed           = 42   
    )
    # For the test data apply only rescale 
    test_data  = test_generator.flow_from_directory(
        directory   = testing_dir,
        target_size = target_image_size,
        color_mode  = color_mode,
        batch_size  = 32, 
        class_mode  = "categorical",
        shuffle     = False,
        seed        = 42
    ) 
    
    model_data = {
        "training_set"    : train_data,
        "validation_set"  : validation_data,
        "testing_set"     : test_data 
    }
    return model_data