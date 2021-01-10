# Import Dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import time
import os
from argparse import ArgumentParser

from classification.model import Submodel_1, Classifier
from segmentation.model import UNet
from utils import ignore_nii, ignore_noncovid, iou_pytorch, convert_to_binary, AttrDict

# Classification Net
import torchvision.models as models


# Functions for training the classifier
def train_net(transfer_net,net, train_data,val_data, batch_size=64, learning_rate=0.01, num_epochs=30, gpu = True):
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be Adam
    learned_parameters = []
    for param in transfer_net.parameters():
        learned_parameters.append(param)
    for param in net.parameters():
        learned_parameters.append(param)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(learned_parameters, lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=True)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    if gpu:
        transfer_net = transfer_net.cuda()
        net = net.cuda()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            if gpu:
              inputs = inputs.cuda()
              labels = labels.cuda()
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(transfer_net(inputs))
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(transfer_net,net, val_loader, criterion,gpu)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)


def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def evaluate(transfer_net, net, loader, criterion, gpu):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()
        outputs = net(transfer_net(inputs))
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss


# plot Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

################################################################################################################################################################################################

# Functions for training the segmentation model
def initialize_loader(train_dataset,valid_dataset,train_batch_size=64, val_batch_size=64):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=val_batch_size,shuffle=True)
    return train_loader, valid_loader

def run_validation_step(args, epoch, model, loader, feature_extractor):

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    losses = []
    ious = []
    with torch.no_grad():
        for i, (images, masks,raw_input) in enumerate(loader):
            if args.gpu:
                images = images.cuda()
                masks = masks.cuda()
                raw_input = raw_input.cuda()
            feature = feature_extractor(raw_input)
            output = model(images.float(),feature)
            # pred_seg_masks = output["out"]

            output_predictions = output.argmax(0)
            loss = compute_loss(output, masks.squeeze(1).long())
            iou = iou_pytorch(output, masks.squeeze(1).long())
            losses.append(loss.data.item())
            ious.append(iou.data.item())

        val_loss = np.mean(losses)
        val_iou = np.mean(ious)
    
    return val_loss, val_iou

def train(args, model, feature_extractor, training_data, valid_data):
    
    # Set the maximum number of threads to prevent crash
    torch.set_num_threads(5)
    # Numpy random seed
    np.random.seed(args.seed)
    
    # Save directory
    # Create the outputs folder if not created already
    save_dir = "outputs/" + args.experiment_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    train_loader, valid_loader = initialize_loader(training_data,valid_data,args.train_batch_size,args.val_batch_size)

    print("Beginning training ...")
    if args.gpu: 
        model.cuda()

    start = time.time()
    trn_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0

    for epoch in range(args.epochs):

        # Train the Model
        model.train() # Change model to 'train' mode
        start_tr = time.time()
        
        losses = []
        for i, (images, masks, raw_input) in enumerate(train_loader):

            if args.gpu:
                images = images.cuda()
                masks = masks.cuda()
                raw_input = raw_input.cuda()
            features = feature_extractor(raw_input)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output = model(images.float(),features)
            # pred_seg_masks = output["out"])
            # _, pred_labels = torch.max(output, 1, keepdim=True)
            loss = compute_loss(output, masks.squeeze(1).long())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())


        # plot training images
        trn_loss = np.mean(losses)
        trn_losses.append(trn_loss)
        time_elapsed = time.time() - start_tr
        print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
                epoch+1, args.epochs, trn_loss, time_elapsed))

        # Evaluate the model
        start_val = time.time()
        val_loss, val_iou = run_validation_step(args, 
                                                epoch, 
                                                model,
                                                valid_loader, feature_extractor)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(save_dir, args.checkpoint_name + '-best.ckpt'))

        time_elapsed = time.time() - start_val
        print('Epoch [%d/%d], Loss: %.4f, mIOU: %.4f, Validation time (s): %d' % (
                epoch+1, args.epochs, val_loss, val_iou, time_elapsed))
        
        val_losses.append(val_loss)
        val_ious.append(val_iou)

    # Plot training curve
    plt.figure()
    # plt.plot(trn_losses, "ro-", label="Train")
    # plt.plot(val_losses, "go-", label="Validation")
    plt.plot(trn_losses,  label="Train")
    plt.plot(val_losses,  label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir+"/training_curve.png")

    # Plot validation iou curve
    plt.figure()
    plt.plot(val_ious, "ro-", label="mIOU")
    plt.legend()
    plt.title("mIOU")
    plt.xlabel("Epochs")
    plt.savefig(save_dir+"/val_iou_curve.png")

    print('Saving model...')
    torch.save(model.state_dict(), os.path.join(save_dir, args.checkpoint_name + '-{}-last.ckpt'.format(args.epochs)))

    print('Best model achieves mIOU: %.4f' % best_iou)


def compute_loss(pred, gt):
    loss = F.cross_entropy(pred, gt,weight=torch.Tensor([0.1,4.15]).cuda())
    return loss



def main(classification_data_path,segmentation_img_path,segmentation_mask_path):

    # Classification
    resnet18 = models.resnet18(pretrained=True) #output in 1x1000

    # Extract features for segmentation input 
    feature_extractor = Submodel_1(resnet18)

    data_path_classify = classification_data_path

    transform_classify = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    data_classify = torchvision.datasets.ImageFolder(root=data_path_classify,transform=transform_classify)

    # Calculate split lengths
    total_size_classify = len(data_classify)
    train_size_classify = round(0.7*total_size_classify)
    valid_size_classify = round(0.15*total_size_classify)
    test_size_classify = round(0.15*total_size_classify)

    # Seperate into Train, Val and Test sets
    random.seed(14)
    train_set_classify, valid_set_classify, test_set_classify = torch.utils.data.random_split(data_classify, [train_size_classify,valid_size_classify,test_size_classify])


    # Initialize handcrafted classifier and train the transfer learning + ANN network
    classifier = Classifier(1000)
    train_net(resnet18,classifier,train_set_classify,valid_set_classify,batch_size=64,learning_rate=0.001,num_epochs=15)
    model_path = get_model_name("classifier", batch_size=64, learning_rate=0.001, epoch=14)
    plot_training_curve(model_path)


    # Report the Test Accuracy on classification
    criterion = nn.BCEWithLogitsLoss()
    gpu = True
    test_err, test_acc = evaluate(resnet18,classifier,torch.utils.data.DataLoader(test_set_classify,batch_size=64),criterion,gpu)
    print('Test Accuracy is',1-test_err)



###########################################################################################################################################################################

    # Segmentation

    data_path_img_autoEncoder = segmentation_img_path
    data_path_mask_autoEncoder = segmentation_mask_path

    transform_autoEncoder = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        ])

    transform_feature_extractor = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        ])

    data_img_feature_extractor = torchvision.datasets.ImageFolder(root=data_path_img_autoEncoder,transform=transform_feature_extractor,is_valid_file=ignore_noncovid)
    data_img_autoEncoder = torchvision.datasets.ImageFolder(root=data_path_img_autoEncoder,transform=transform_autoEncoder,is_valid_file=ignore_noncovid)
    data_mask_autoEncoder = torchvision.datasets.ImageFolder(root=data_path_mask_autoEncoder,transform=transform_autoEncoder,is_valid_file=ignore_nii)

    print('the number of images match=',len(data_img_autoEncoder) == len(data_mask_autoEncoder))

    # Build (image,Mask) pairs
    data_autoEncoder = []
    for i in range(len(data_img_autoEncoder)):

        data_autoEncoder.append((data_img_autoEncoder[i][0],data_mask_autoEncoder[i][0],data_img_feature_extractor[i][0]))


    # Training:Validation:Test = 0.7:0.15:0.15
    random.seed(14)
    random.shuffle(data_autoEncoder)

    train_index = int(len(data_autoEncoder) * 0.7)
    val_index = int(len(data_autoEncoder) * 0.85)

    training_data = data_autoEncoder[:train_index]
    valid_data = data_autoEncoder[train_index:val_index]
    test_data = data_autoEncoder[val_index:]
    print("# Train Set: " + str(len(training_data)))
    print("# Test Set: " + str(len(test_data)))
    print("# Val Set: " + str(len(valid_data)))


    # Unet Hyperparameters
    args_unet = AttrDict()

    args_dict = {
                'gpu':True, 
                'checkpoint_name':"unet_segmentation", 
                'learn_rate':0.01, 
                'train_batch_size':128, 
                'val_batch_size': 256, 
                'epochs':20, 
                'seed':14,
                'experiment_name': 'unet_segmentation',
    }
    args_unet.update(args_dict)


    # Train Unet
    unet = UNet(10,2,1)
    feature_extractor = feature_extractor.cuda()
    train(args_unet,unet,feature_extractor,training_data,valid_data)


    # Visualize a few predictions in test set
    feature_extractor = feature_extractor.cuda()
    for (imgs,masks,raw_input) in test_data:
        pred = unet(imgs.float().cuda(),feature_extractor(raw_input.cuda()))
        msk = masks
        raw = raw_input
        pred = torch.argmax(pred, 1)
        break

    for i in range(len(pred)):
        fig = plt.figure(figsize=(15,4.5))
        plt.title('prediction vs. ground truth vs. input image')
        ax = fig.add_subplot(1,3,1)
        plt.imshow(pred[i+10].cpu().detach().numpy())
        ax = fig.add_subplot(1,3,2)
        plt.imshow(msk[i+10].cpu().detach().numpy().squeeze(0))
        ax = fig.add_subplot(1,3,3)
        fig.savefig("segmentation" + str(i) + ".jpeg")
        i += 1
        if i > 10:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification and Segmentation Model')
    parser.add_argument('classification_data_path', help='path to classification data folder')
    parser.add_argument('segmentation_img_path', help='path to segmentation images folder')
    parser.add_argument('segmentation_mask_path', help='path to segmentation masks folder')

    args = parser.parse_args()

    main(args.classification_data_path,args.segmentation_img_path,args.segmentation_mask_path)


