import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data_loaders
from models import FCNet, ConvNet
import matplotlib.pyplot as plt

### Fixed variables ###
BATCH_SIZE = 1000
SEED = 42
#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

### Setup seeds for deterministic behaviour of computations ###
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
###############################################################

# Load dataset
CIFAR_10_dataset = get_data_loaders(BATCH_SIZE)
    
def train_and_test(model_name, dataset, num_epochs, learning_rate, activation_function_name):
    if model_name == "fcnet":
        model = FCNet(activation_function_name=activation_function_name).to(device)
    elif model_name == "convnet":
        model = ConvNet(activation_function_name=activation_function_name).to(device)
    else:
        raise Exception("No such model. The options are 'fcnet' and 'convnet'.")
    train_loader = dataset[0]
    test_loader = dataset[1]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    test_accuracy = None
    # TODO: Train the network
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # back propogation
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = (100* correct) / total
    return model, test_accuracy

def hyperparameters_grid_search(model_name, dataset):
    #print('got to it')
    learning_rate_options = [1e-7, 1e-3, 1]
    activation_function_name_options = ["sigmoid", "relu"]
    best_test_accuracy = 0
    best_hyperparameters = {"learning_rate": None, "activation_function_name": None}
    for lr in learning_rate_options:
        #print('in for loop')
        for activation_function in activation_function_name_options:
            model, test_accuracy = train_and_test(model_name, dataset, num_epochs=5, learning_rate=lr, activation_function_name=activation_function)
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_hyperparameters = {"learning_rate": lr, "activation_function_name": activation_function}

            print(f"Current hyperparameters: num_epochs=5, learning_rate={lr}, activation_function_name={activation_function}")
            print(f"Current accuracy for test images: {test_accuracy}%")
            print(f"Best test accuracy: {best_test_accuracy}%")
            print("Best hyperparameters:", best_hyperparameters)

if __name__ == "__main__":
    model, test_accuracy = train_and_test(model_name="fcnet", dataset=CIFAR_10_dataset, num_epochs=5, learning_rate=1e-3, activation_function_name="relu")
    print(f"FCN Test accuracy is: {test_accuracy}%")
    torch.save(model.state_dict(), "cifar-10-fcn.pt")
    print("FCN model saved.")

    # Plot images with predictions
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #print('got to line 1')
    images, labels = next(iter(CIFAR_10_dataset[1]))
   # print('got to line 2') 
    outputs = model(images.to(device))
    #print('got to line 3')
    _, predicted = torch.max(outputs, 1)
    #print('got to line 4')
    fig = plt.figure("Example Predictions", figsize=(12, 5))
    #print('got to line 5')
    fig.suptitle("Example Predictions")
    for i in range(5):
    #    print('got to loop, iteration', i)
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        img = images[i] * 0.25 + 0.5     # unnormalize
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"Label: {classes[labels[i]]}\nPredicted: {classes[predicted[i]]}")
    # print('got to line 6')
    plt.show()
    #print('got to line 7')
    # Train and test convolutional neural network and save the model
    print('going into test CNN')
    model, test_accuracy = train_and_test(model_name="convnet", dataset=CIFAR_10_dataset, num_epochs=5, learning_rate=1e-3, activation_function_name="relu")
    print(f"CNN Test accuracy is: {test_accuracy}%")
    torch.save(model.state_dict(), "cifar-10-cnn.pt")
    print("CNN model saved.")

    # Do hyperparameter search
    hyperparameters_grid_search("convnet", CIFAR_10_dataset)
