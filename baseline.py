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

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from utils import ignore_nii, ignore_noncovid, iou_pytorch, convert_to_binary, AttrDict


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

def train(args, model, feature_extractor):
    
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

class baseline_autoEncoder(nn.Module):
    def __init__(self):
        super(baseline_autoEncoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.Conv2d(128, 256, 7),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main(classification_data_path,segmentation_img_path,segmentation_mask_path):
    # Classification Baseline
    data_path_rndForest = classification_data_path

    transform_rndForest = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_rndForest = torchvision.datasets.ImageFolder(root=data_path_rndForest,transform=transform_rndForest)

    # Calculate split lengths
    total_size_rndForest = len(data_rndForest)
    train_size_rndForest = round(0.7*total_size_rndForest)
    test_size_rndForest = round(0.3*total_size_rndForest)

    train_set_rndForest, test_set_rndForest = torch.utils.data.random_split(data_rndForest, [train_size_rndForest,test_size_rndForest] )


    # Seperate random Forest Data into images and labels
    train_set_imgs_rndForest = []
    train_set_labels_rndForest = []
    test_set_imgs_rndForest = []
    test_set_labels_rndForest = []
    count = 1
    for data in train_set_rndForest:
        clear_output(wait=True)
        print("Current random Forest split progress:", count, "/",len(data_rndForest))
        train_set_imgs_rndForest.append(data[0])
        train_set_labels_rndForest.append(data[1])
        count += 1

    for data in test_set_rndForest:
        clear_output(wait=True)
        print("Current random Forest split progress:", count, "/",len(data_rndForest))
        test_set_imgs_rndForest.append(data[0])
        test_set_labels_rndForest.append(data[1])
        count += 1

    train_set_imgs_rndForest = torch.stack(train_set_imgs_rndForest)
    train_set_labels_rndForest = np.array(train_set_labels_rndForest)
    test_set_imgs_rndForest = torch.stack(test_set_imgs_rndForest)
    test_set_labels_rndForest = np.array(test_set_labels_rndForest)


    # Classification Baseline Model
    # 1-Random Forest (From Scikit Learn)
    baseLine_rndForest = RandomForestClassifier(max_depth=2, random_state=0)
    # Train randomForest
    with torch.no_grad():
        trainFeature = resnet18(train_set_imgs_rndForest)
        testFeature = resnet18(test_set_imgs_rndForest)

    baseLine_rndForest.fit(trainFeature, train_set_labels_rndForest)

    # Test randomForest Accuracy
    pred = baseLine_rndForest.predict(testFeature)
    correct = (pred == test_set_labels_rndForest).sum()
    total = testFeature.shape[0]
    print("Random Forest Accuracy is: ", correct/total*100)


    #############################################################################################################################################
    # Segmentation Baseline Model
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

    # Baseline Hyperparameters
    args_baseline = AttrDict()
    args_dict = {
                'gpu':True, 
                'checkpoint_name':"baseline_segmentation", 
                'learn_rate':0.1, 
                'train_batch_size':64, 
                'val_batch_size': 256, 
                'epochs':10, 
                'seed':0,
                'experiment_name': 'baseline_segmentation',
    }
    args_baseline.update(args_dict)

    # Train baseline model
    baseline_segmentation = baseline_autoEncoder()
    train(args_baseline,baseline_segmentation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification and Segmentation Baseline Model')
    parser.add_argument('classification_data_path', help='path to classification data folder')
    parser.add_argument('segmentation_img_path', help='path to segmentation images folder')
    parser.add_argument('segmentation_mask_path', help='path to segmentation masks folder')

    args = parser.parse_args()

    main(args.classification_data_path,args.segmentation_img_path,args.segmentation_mask_path)

