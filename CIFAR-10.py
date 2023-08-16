# Importing required libraries
from fastai.vision.all import *

# Function to load CIFAR-10 data
def get_data():
    path = untar_data(URLs.CIFAR)
    dls = ImageDataLoaders.from_folder(path, valid='test', item_tfms=Resize(224), batch_tfms=Normalize.from_stats(*cifar_stats))
    return dls

# Function to create and train the model
def train_model(dls):
    model = cnn_learner(dls, resnet34, metrics=accuracy)
    model.fine_tune(5)
    return model

# Main function to run the entire program
def main():
    dls = get_data()
    model = train_model(dls)

    # Displaying training results
    model.show_results()

    # Save the model
    model_name = "cifar10_model"
    model.save(model_name)
    print(f"Model '{model_name}' saved successfully.")

# Entry point
if __name__ == "__main__":
    main()
