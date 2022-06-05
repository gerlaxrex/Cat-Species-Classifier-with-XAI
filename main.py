from model.model_training import train_model, prepare_model
from data.datasets import preprocess_dataset, preprocessed_images, labels, build_dataset
from sklearn.model_selection import train_test_split
from configs import *

if __name__ == '__main__':
    # Prepare the datasets for training and validation
    img_dir = './Felidae'
    preprocess_dataset(img_dir)

    # Train/Valid split
    train_images, valid_images, train_labels, valid_labels = train_test_split(preprocessed_images,
                                                                              labels,
                                                                              train_size=0.8,
                                                                              stratify=labels)

    # Build the datasets
    train_ds, train_size = build_dataset(train_images, train_labels, TRAIN_BS)
    valid_ds, valid_size = build_dataset(valid_images, valid_labels, VALID_BS, shuffle=False)

    # Define the Steps for training and validation
    SPE_TRAIN = train_size // TRAIN_BS
    SPE_VALID = valid_size // VALID_BS

    # Define, build and compile the model
    cat_species_classifier = prepare_model()

    # Train the model
    cat_species_classifier, history = train_model(cat_species_classifier, train_ds, valid_ds, SPE_TRAIN, SPE_VALID)



