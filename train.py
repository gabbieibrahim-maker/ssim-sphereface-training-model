# Angular Contrastive Learning
# Sphereface / A-softmax loss

numOfEpochs = 100 # choose number of epochs
numOfBatches = 16 # choose number of batches
encoderNames = ["dinov2", "dinov3", "resnet34", "resnet50"]
encoderName = encoderNames[3] # choose encoder
balanceByOptions = ["anatomy", "anatomy_artifact"] # grouping for a balanced group for having the same number of samples per epoch
balanceByDistribution = balanceByOptions[0]
difference_in_ssim = 0.05 # choose how many classes you want (1 / difference_in_ssim)
learning_rate = 0.0001 #3e-4 recommended -> update: val loss was increasing very early
optimizerWillUse = ['SGD', 'Adam']
optimizerIndex = optimizerWillUse[1]
multiplier = 0.01 # penalty multiplier for changing lambda in loss function; default 0.1, experimented with 0.5, but model gets worse on epoch 3, meaning angular margin penalty is getting too harsh
angularMargin = 5 # default 4; for AngleLinear
loRARank = 16 # common values 16, 32, 64
smallSampleSize = None # keep None for regular
numOfWorkers = 3 # more workers makes model run faster

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
# from hugging face; AutoModel handles ViTs (dino) and Resnets differently

from datasetgabbie import getloader_3d_patches
import torch.optim as optim
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/scratch/slurm-{os.environ.get('SLURM_JOB_ID', 'local')}/torch_cache_{os.getpid()}"
import wandb # used to log the metrics of our model

from torch.nn import Parameter
import math
from peft import LoftQConfig, LoraConfig, get_peft_model

import time
script_start_time = time.time()

dictionaryOfHyperparameters = {
    "encoder" : encoderName,
    "learning rate" : learning_rate,
    "optimizer" : optimizerIndex,
    "difference in SSIM" : difference_in_ssim,
    "angular margin in AngleLinear" : angularMargin,
    "loRA Rank" : loRARank,
    "penality multiplier in AngleLoss" : multiplier,
    "number of batches" : numOfBatches,
    "number of epochs" : numOfEpochs,
    "sample size" : smallSampleSize,
    "number of workers" : numOfWorkers,
    "balance by" : balanceByDistribution
}

# to track on Weights & Biases
wandb.init(
    entity = 'gabbieibrahim07-carnegie-mellon-university',
    project = 'sphereface training and validation testing',
    name = 'whole dataset training - 8.17.26 changing gamma to 2; balance_by anatomy', # if we use same name, it will overwrite stuff with this name
    config = dictionaryOfHyperparameters,
    save_code = True
)

plottingVisualEmbeddings = False # variable to control if plotting embeddings

# https://arxiv.org/pdf/1704.08063 --> sphereface article for face recognition
# https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py --> sphereface sample code; large amount of code is pasted from here

def myphi(x,m): # calculates cos(m*theta) (sphereface)
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = angularMargin, phiflag=True): # m constitutes the size of the angular margin. the larger m is, the tighter features will be clustered
        super(AngleLinear, self).__init__()
        self.in_features = in_features # input image (vector with 512 features)
        self.out_features = out_features # classification (1 / 0.05 seperation margin) = 20 classes
        self.weight = Parameter(torch.Tensor(in_features,out_features)) # creating a 2D tensor (matrix) with the 512 features and 20 classes
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5) # initializing our weights randomly
        self.phiflag = phiflag # to use the phiflag formula, all in the myphi helper function
        self.m = m # margin value
        self.mlambda = [ # will use whatever index m is for faster calculation of cos(m*theta)
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len         # the 512 image features from the encoder
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features     # vectors for the 20 classes

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B         # length of the image embeddings
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum    # length of the class weights

        # finds dot product between the image vectors and the 20 weights vectors

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta) # calculates the cos(m*theta)
            theta = cos_theta.detach().acos() # inverse cosine to find the angle between the image and weights
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        # rescales everything
        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)        # returns the distance and the penalized margin distance


class AngleLoss(nn.Module): # sphereface / A-softmax (angular)
    def __init__(self, gamma=2): # gamma: focusing parameter; (focal loss)
        # scales the loss contribution of individual samples based on how well the model already classifies them
        # focuses on hard examples where maybe there are fewer training samples for that example
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0 # will decrease the more you train; lambda in annealing optimization strategy for A-softmax loss (in article)

    def forward(self, input, target):
        if self.training:
          self.it += 1 # for each time the model updates, it increases by 1 for the self.lamb calculation; does not update if testing
        cos_theta,phi_theta = input # both types are torch.Tensor
        target = target.to(cos_theta.device) # saves target to same device that cos_theta loss is saved to (to stay on gpu)
        # {cos_theta.shape} --> [1, 20]

        # the following helps us isolate the correct target class so we can apply a harsher angular margin only to the correct class;
        # it leaves the incorrect ones (classes with 0's) alone
        # this harsh anguar margin helps us classify more distinctly, based on how big "m" parameter is
        # If our penalty multiplier m = 4, the model has to make the angle to the correct class 4 times smaller
        # than the angle to any other class before it stops getting penalized

        target = target.view(-1,1) #size=(B,1) # column vector for classes 0-19. target tensor gets reshaped into 2d array
        scatterParameterIdx = a = target.detach().view(-1,1)
        scatterIdx = a.to(torch.int64) # needs dtype int32/int64
        index = torch.zeros_like(cos_theta, dtype = torch.bool) #size=(B,Classnum) # creates a matrix of 0's that is the same shape as cos_theta
        # print(f'scatter index shape {scatterIdx.shape}') --> [4, 1]
        index.scatter_(1,scatterIdx,1) # (dimension 1 / columns, which indices recieve new vals, new scaler value) puts a 1 at the correct classification, and 0's for the rest

        index = index.bool() # masked fill only supports boolean masks

        # all apart of the annealing optimization strategy for A-softmax loss (in article)
        # multiplier affects how long self.lamb takes to drop down to LambdaMin
        # multiplier was set to 0.1 in sample code (initialized it at top of file)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+multiplier*self.it )) # only changes when training (self.it changes); regularization parameter - controls how much penalty is added to model
        if self.training and self.it % 500 == 0:
            wandb.log({"lambda_anneal": self.lamb, "iteration": self.it})
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        targetInt = target.to(torch.int64)
        logpt = F.log_softmax(output, dim = 1) # converts into probabilities
        logpt = logpt.gather(1,targetInt) # gets the class with highest probability (dimension, index)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class EncoderAngularRegressor(nn.Module):
    # all encoder names (["dinov2", "dinov3", "resnet34", "resnet50"]) use this format
    """
    encoder backbone + MLP head for scalar regression.

    Input:
        pixel_values: [B, 3, H, W] (batches, channels, height, width)

    Output:
        pred: [B, 512]  vector that will be used for the angular prediction of where it goes on hyphersphere (batches, num of features)
    """

    def __init__(
        self,
        diffInSSIM,
        encoderType,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        pooling: str = "cls",  # "cls" or "mean_patch"
    ):
        super().__init__()

        if encoderType == 'dinov2':
            modelName = "facebook/dinov2-base"
        elif encoderType == 'dinov3':
            modelName = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        elif encoderType == 'resnet34':
            modelName = "microsoft/resnet-34"
        elif encoderType == 'resnet50':
            modelName = "microsoft/resnet-50"

        # LoRA - low rank adaptation
        # freezes base mode and injects small trainable matrices, so only these matrices get updated
        # same performance as full tuning, but only train a small portion of the parameters
        # common rank sizes: 16, 32, 64 (higher -> more learning capacity, more memory)

        # code from https://huggingface.co/docs/peft/main/en/conceptual_guides/lora

        baseModel = AutoModel.from_pretrained(modelName)
        if "resnet" in encoderType:
            target_mods = ["convolution"] # Conv2d layers
            init_method = True
            loftq_cfg = {}
        else:
            target_mods = "all-linear"    #  inear layers (for DINO)
            init_method = "loftq"
            loftq_cfg = LoftQConfig(loftq_bits=4) #LoRA fine tuning quantization: bit width for quantization (convert from high to low precision nums) is 4 bits
        lora_config = LoraConfig(                           # LoRA configuration class
                                r = loRARank,                 # rank
                                lora_alpha=32,              # constant scaling hyperparameter; using rank * 2
                                target_modules=target_mods, # which layers recieve the trainable LoRA matrices
                                init_lora_weights=init_method, # how it initializes layers
                                loftq_config=loftq_cfg, # quantization bit width
                            )
        peft_model = get_peft_model(baseModel, lora_config)

        peft_model_gpu = peft_model.cuda()

        self.backbone = peft_model # our backbone is now the LoRA model, not just the original baseModel

        self.pooling = pooling
        self.diffInSSIM = diffInSSIM

        numOfClasses = round(1 / self.diffInSSIM)
        self.numOfClasses = numOfClasses

        self.numOfFeatures = 512 #  # https://arxiv.org/pdf/1704.08063   --> model contained 512 fully connected layers

        # transformers (dino) have a single hidden dimension across all layers
        # resnets have dimensions that change sizes between each layer (spatial resolution dec while channels inc)

        if hasattr(self.backbone.config, "hidden_size"): # for dino/ViT models
            embed_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "hidden_sizes"): # for resnet models
            embed_dim = self.backbone.config.hidden_sizes[-1]

        # embed_dim is the vector of the image that the encoder outputs after passing through the model

        # reduce the encoder dimensions to 512 layers
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.numOfFeatures),  # outputs the features to represent the spatial coordinates of the image for the hypersphere

        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifyClass = AngleLinear(self.numOfFeatures, self.numOfClasses) # this takes the vector of the image (with self.numOfFeatures)
        # and helps classify where it goes on the hypersphere relative to the number of classes

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        # ViTs: outputs.last_hidden_state: [B, 1 + num_patches, D]  [batch, sequence, dimension]  (3D)

        # resnet outputs 2D feature maps; last_hidden_state: [B, C, H, W] (4D)
        # use pooler_output from Hugging Face (Global Average Pooling)

        # tokens variable: tensor (broader representation of vectors/scalers/matrices)
        # that represents the final output from the encoder model
        # tokens are chunks of data that the model processes

        # extracting following if/else from gemini
        if "resnet" in self.backbone.config.model_type.lower():
            tokens = outputs.last_hidden_state # [B, C, H, W]
            feat = tokens.mean(dim=[2,3]) # [B, C]
        elif "dino" in self.backbone.config.model_type.lower():
            tokens = outputs.last_hidden_state # [B, 1 + num_patches, D]
            if self.pooling == "cls": # vector trained to represent the entire input (learnable vector, not fixed)
              feat = tokens[:, 0]   # CLS (classification) token: [B, D] --> extracts the first column (index 0)
            elif self.pooling == "mean_patch": # averages all the embeddings of all the tokens in the sequence
              feat = tokens[:, 1:].mean(1) # mean over patch tokens: [B, D]

        pred = self.head(feat) # outputs the self.numOfFeatures
        output = self.classifyClass(pred) # turns features into coordinates relative to num of classes

        if plottingVisualEmbeddings:
            return pred

        return output


'''

using MNIST training and testing template (from google colab)

'''

network = EncoderAngularRegressor(difference_in_ssim, encoderName) # using a step size of 0.05 to divide up the ssim scores into classes
network = torch.compile(network) # torch.compile will fuse a lot of the operations in the neural network and becomes more optimized
network = network.cuda()

if optimizerIndex == 'SGD':
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
elif optimizerIndex == 'Adam':
    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate, weight_decay = 5e-4)

scaler = torch.amp.GradScaler('cuda') # mix precision: accelerates model training and reduces memory usage

# sphereface paper updates the learning rate after 10-20k iterations
# scheduler updates learning rate based on training process and results. deciding to use ReduceLROnPlateau method
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, 
                                        mode = 'min', # means that we want to watch the loss, where lower is better,
                                        factor = 0.5, # reduces learning rate by 0.5 every time this is triggered
                                        patience = 5 # wait 5 epochs of no improvement before changing the learning rate
                                        )      
loss_function = AngleLoss()


data_root = "/vast/tibrahim/jil202/data"
anatomies = [ "knee", "brain", "prostate"]

classDistributionDictionary = dict()
# balance_by is dictated by "anatomy" right now; checking if class distribution is even

def train(epoch):

    loss_function.train() # for Angle Loss class
    network.train()

    loss_sum, distance_sum = 0, 0
    totalNumOfImagesSeen = 0
    totalSamples = len(train_set)
    if totalSamples < 10:
      display = 1
    else:
      display = totalSamples // 10

    # print(f'length of train set is {len(train_set)}') --> 881

    for index, info in enumerate(train_set, start = 1):
        condition, target, prompt, target_type, anatomy, artifact, ssimCalc, validSSIM = info

        condition = condition.cuda()

        floatSSIM = getValidFloatSSIM(ssimCalc, validSSIM)
        # floatSSIM is now a tensor with size of the batch size minus the number of invalid SSIMs. size is <= batch size
        targetIndexVector = findTargetIndices(floatSSIM).cuda() 
        # a tensor with size of the batch size minus the number of invalid SSIMs. size is <= batch size. instead of values being SSIMs, they are now classes

        printAnatomy = anatomy[0]

        numOfSlices = condition.shape[4] # 16

        for j in range(numOfSlices):
            conditionSingleSlice = condition[:, :, :, :, j] # will evaluate the same slice for all images in the batch
            # this variable contains all batches. shape: [4, 1, 96, 96]
            totalNumOfImagesSeen += targetIndexVector.size(0) # targetIndexVector.size(0) is the batch size
            with torch.amp.autocast('cuda'): # forward pass
                predictionOutput = getPredictionOutput(info, conditionSingleSlice)
                if predictionOutput is None:
                    continue
                #   print(f'num of batches {numOfBatches}')
                cosOutput = predictionOutput[0] # --> cos_theta)
                _, predicted = torch.max(cosOutput.detach(), 1) # find the index of the class value (highest probability)

                loss = loss_function(predictionOutput, targetIndexVector)

                current_distance = torch.abs(predicted.cpu() - targetIndexVector.cpu()).float()

            actual_loss = loss.item()
            current_distance_batch = current_distance.mean().item() # averages all distance within the valid vatch
            current_distance_sum = current_distance.sum().item() # sums all the batches distances (cannot safely say each sum the same size tensors since some get filtered by validSSIM)

            # scale the loss before backpropogation for GradScaler
            scaled_loss = scaler.scale(loss) # inflated loss
            optimizer.zero_grad()
            scaled_loss.backward() # all the gradients also get inflated by the same amount the loss did;
            # scaled is not used in test function since .backward() doesn't happen;
            # scaled is used to make .backward() safer to prevent small gradients from rounding to 0

            # update weights
            # optimizer.step() instead of this, use scaler to update the weights
            scaler.step(optimizer) # rescales the gradients back down before updating the actual weights
            scaler.update() # changes the scaling factor based on if gradient turned into any inf or nan values

            loss_sum += actual_loss
            distance_sum += current_distance_sum

            curr_avg_loss = loss_sum / totalNumOfImagesSeen
            curr_avg_distance = distance_sum / totalNumOfImagesSeen

            if index % display == 0 and j == (numOfSlices - 1):
                print(f"{encoderName}, Train Epoch {epoch}, Anatomy: {printAnatomy}, Slice: {j + 1}, train loss: {actual_loss:.4f}, class distance: {current_distance_batch}")
                wandb.log({
            'checkpoint': 'train intermediate steps',
            "epoch": epoch,
            "anatomy": anatomy,
            "training loss": actual_loss,
            "class error when training": current_distance_batch
            })
        
        # for checking class distribution
        if epoch == 1:
            for classNum in targetIndexVector:
                classNumVal = int(classNum.item()) # need .item() since targetIndexVector is a PyTorch tensor
                classDistributionDictionary[classNumVal] = classDistributionDictionary.get(classNumVal, 0) + 1


    avg_train_loss = loss_sum / totalNumOfImagesSeen
    avg_distance_train = distance_sum / totalNumOfImagesSeen
    torch.save(network.state_dict(), "./model.pt") # saves the weights of the model after every batch (but we can change this to save after every epoch if we want)

    # checking class distribution
    # if epoch == 1:
    #     print("\n class distribution for full dataset \n")
    #     for k in sorted(classDistributionDictionary.keys()):
    #         print(f"Class {k + 1}: {classDistributionDictionary[k]} samples")
    #     print("\n")
    
    #  class distribution for full TRAINING dataset with balance_by = "anatomy"
    # --> results: heavily skewed. options: change the balance_by or change penalty when class number is low and model gets it incorrect

    # Class 1: 1 samples
    # Class 2: 4 samples
    # Class 3: 4 samples
    # Class 4: 6 samples
    # Class 5: 8 samples
    # Class 6: 8 samples
    # Class 7: 22 samples
    # Class 8: 17 samples
    # Class 9: 28 samples
    # Class 10: 49 samples
    # Class 11: 67 samples
    # Class 12: 90 samples
    # Class 13: 110 samples
    # Class 14: 102 samples
    # Class 15: 139 samples
    # Class 16: 192 samples
    # Class 17: 261 samples
    # Class 18: 296 samples
    # Class 19: 283 samples
    # Class 20: 294 samples

    # in reality, the model has high confidence (assigns higher probability) to higher classes and probably assigns a very low probability to low classes. when this happens and the reality is a low class number, loss is very high

    
    # class distribution with balance_by = "anatomy_artifact"

    # Class 3: 1 samples
    # Class 6: 2 samples
    # Class 7: 8 samples
    # Class 8: 6 samples
    # Class 9: 6 samples
    # Class 10: 12 samples
    # Class 11: 25 samples
    # Class 12: 28 samples
    # Class 13: 43 samples
    # Class 14: 36 samples
    # Class 15: 44 samples
    # Class 16: 59 samples
    # Class 17: 78 samples
    # Class 18: 107 samples
    # Class 19: 95 samples
    # Class 20: 129 samples

    print(f'\nDone Training, Anatomy: {printAnatomy}, Epoch {epoch}\n')
    print(f'\nLoss: {avg_train_loss:.4f}, Accuracy: None, Distance: {avg_distance_train:.4f}')
    return avg_train_loss, avg_distance_train


def test(epoch):

    loss_function.eval()
    network.eval()
    correct, totalNumOfImagesSeen = 0, 0
    loss_sum, accuracy_sum, distance_sum = 0,0,0
    totalClassDistance = 0
    totalSamples = len(val_set)

    if totalSamples < 10:
      display = 1
    else:
      display = totalSamples // 10

    with torch.no_grad():
        for index, info in enumerate(val_set, start = 1):
            condition, target, prompt, target_type, anatomy, artifact, ssimCalc, validSSIM = info

            condition = condition.cuda()

            floatSSIM = getValidFloatSSIM(ssimCalc, validSSIM)
            targetIndexVector = findTargetIndices(floatSSIM).cuda()
            # target should be a vector (the size of the number of batches) that indicate the index/class number for the target (floatSSIM)

            numOfSlices = condition.shape[4] # 16

            printAnatomy = anatomy[0]


            for j in range(numOfSlices):
                conditionSingleSlice = condition[:, :, :, :, j] # will evaluate the same slice for all images in the batch
                # this variable contains all batches. shape: [4, 1, 96, 96]

                with torch.amp.autocast('cuda'):
                    predictionOutput = getPredictionOutput(info, conditionSingleSlice)
                    if predictionOutput is None:
                        continue

                    # output of the Angle Linear forward function is this:
                    # output = (cos_theta,phi_theta)
                    # phi_theta is used for the Angle Loss function (angular margin penalty used in training, not testing)

                    # use cos_theta to see the pure, unpenalized angle
                    cosOutput = predictionOutput[0] # --> cos_theta)
                    _, predicted = torch.max(cosOutput.detach(), 1) # find the index of the class value (highest probability)

                    # loss_sum += loss_function(predictionOutput, targetIndexVector)
                    correct = predicted.eq(targetIndexVector).cpu().sum().item() # float
                    totalNumOfImagesSeen += targetIndexVector.size(0) # targetIndexVector.size(0) is the batch size

                    current_distance = torch.abs(predicted.cpu() - targetIndexVector.cpu()).float()

                    loss = loss_function(predictionOutput, targetIndexVector)

                current_loss = loss.item()
                current_distance_batch = current_distance.mean().item() # float
                current_distance_sum = current_distance.sum().item()
                current_batch_size = targetIndexVector.size(0)
                current_accuracy = (correct / current_batch_size) 

                loss_sum += current_loss
                accuracy_sum += correct
                distance_sum += current_distance_sum

                current_avg_loss = loss_sum / totalNumOfImagesSeen
                current_avg_accuracy = accuracy_sum / totalNumOfImagesSeen
                current_avg_distance = distance_sum / totalNumOfImagesSeen

                if index % display == 0 and j == (numOfSlices - 1):
                    print(f"{encoderName}, Test Epoch {epoch}, Anatomy: {printAnatomy}, Slice: {j + 1}, val loss: {current_loss:.4f}  val accuracy: {current_accuracy:.4f}, class distance: {current_distance_batch:.2f}")
                    wandb.log({
            'checkpoint': 'val intermediate steps',
            "epoch": epoch,
            "anatomy": anatomy,
            "validation loss": current_loss,
            "class error when testing": current_distance_batch,
            "validation accuracy": current_accuracy
            })

    avg_val_loss = loss_sum / totalNumOfImagesSeen
    avg_val_accuracy = accuracy_sum / totalNumOfImagesSeen
    avg_distance_test = distance_sum / totalNumOfImagesSeen
    print(f'\nDone Testing, Anatomy: {printAnatomy}, Epoch {epoch}\n')
    print(f'\nLoss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}, Distance: {avg_distance_test:.4f}')
    return avg_val_loss, avg_val_accuracy, avg_distance_test


def findTargetIndices(target): # target is the tensor of the float SSIM values for the whole batch
    stepCt = network.diffInSSIM
    numOfClasses = network.numOfClasses
    classNum = torch.ceil(target/stepCt) - 1 # index 0 is [0, 0.05), index 1 is [0.05, 0.1), etc
    # classNum is a tensor with size of the batch size minus the number of invalid SSIMs. size is <= batch size. instead of values being SSIMs, they are now classes
    classNumClamped = torch.clamp(classNum, min = 0, max = numOfClasses - 1)
    return classNumClamped.to(torch.int64)



def getRGB(inputImgCuda):

    x = inputImgCuda.float()
    # normalize
    B = x.shape[0] # scales each pixel to be between 0 and 1 (normalized)
    # x.view(B, -1) flattens from 96x96 image to a 1D vector (temporarly [B, 1 96, 96] -> [B, 9216])
    # finding lowest and highest pixel value in the image
    # .view(B, 1, 1, 1) reshapes single val back into 4D tensor
    x_min = x.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    x_max = x.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    x = (x - x_min) / (x_max - x_min + 1e-8) # min max formula to rescale data between 0 and 1

    # convert 1 channel (grayscale) to 3 channels (RGB)
    x = x.repeat(1, 3, 1, 1) # 1 batch, 3 channels, 1 height, 1 width
    # (repeat 1, 3, 1, 1, times; keeps batch size, height, width the same since * 1)
    # [B, 1, 96, 96] -> [B, 3, 96, 96] since models require rgb images

    # resize
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False) # PyTorch equivalent of cv2.resize()
    # [B, 3, 96, 96] -> [B, 3, 224, 224]

    # for ImageNet normalization
    device = next(network.parameters()).device
    x = x.to(device)

    # values for rgb images in the ImageNet dataset
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    # mean and std
    x = (x - mean) / std
    return x



def getValidFloatSSIM(ssimCalc, validSSIM):
    # validSSIM is a tensor of True's and False's
    ssimCalcValid = torch.masked_select(ssimCalc, validSSIM) # this works since ssimCalc is same num of dimensions as valid SSIM
    floatSSIM = ssimCalcValid.float()
    # ssimCalcValid/floatSSIM is now a tensor with size of the batch size minus the number of invalid SSIMs. size is <= batch size
    return floatSSIM


def getPredictionOutput(info, conditionSingleSlice):
    condition, target, prompt, target_type, anatomy, artifact, ssimCalc, validSSIM = info

    # ssimCalc and validSSIM are both tensors with size of the batch size

    floatSSIM = getValidFloatSSIM(ssimCalc, validSSIM)

    # distHypersphere = LpDistance(p=2, power=1, normalize_embeddings=True) # finds the geometric distance on the hypersphere with normalized vectors

    # "data" is the slice of data with some type of artifact (with size [1, 1, 96, 96, 16]) # batches, channels, height, width, slices
    # "target" is the ssim calculation, size [1]

    # {validSSIM.shape} --> [4]
    # {condition.shape} --> [4, 1, 96, 96, 16]

    mask = (validSSIM == True)
    validSingleConditionSlice = conditionSingleSlice[mask] # filters along index 0.
    # Have to do this since condition shape has 5 dimensions, but validSSIM has 1. shape: [4, 1, 96, 96, 16]

    if validSingleConditionSlice.shape[0] == 0:
        return None

    # current shape [B, 1, 96, 96], need [B, 3, 224, 224]
    x = validSingleConditionSlice.float() # convert images into floats because neural network weights are floats
    x = getRGB(x) # [B, 3, 224, 224]

    # forward pass
    predictionOutput = network(x) # "data" -> [B, 3, 224, 224]
    # predictionOutput is (cos_theta,phi_theta) from AngleLoss
    return predictionOutput

all_val_sets = dict()

for epoch in range(1, numOfEpochs + 1): # each epoch will be trained and tested
    totalValLossPerEpoch = 0

    for anatomy in anatomies:
        print(f"\n--- Epoch {epoch}, loading data for anatomy: {anatomy} ---")
        # from datasetgabbie.py file (/ihome/tibrahim/rflab)yes 
        train_set, val_set = getloader_3d_patches(
                batch_size = numOfBatches, # common values: 32, 64
                data_root = data_root,
                contrast = anatomy,
                sample = 1.0,
                num_workers = numOfWorkers,
                distributed = False,
                rank = 0,
                world_size = 1,
                train_shuffle = None, # if samples per contrast is not None, decide if you want the same samples or random samples each epoch
                samples_per_contrast = smallSampleSize, # controls how many samples we get for each anatomy
                balance_by = balanceByDistribution,
                patch_shape=(96, 96, 16),
                augment=None,
                augment_kwargs=None,
                artifacts=None,
            )
        print(f'train batches: {len(train_set)}, val batches: {len(val_set)}')

        all_val_sets[anatomy] = val_set # only records the last validation set for each anatomy; others get overwritten

        avg_train_loss, avg_distance_train = train(epoch)
        avg_val_loss, avg_val_accuracy, avg_distance_test = test(epoch)
        totalValLossPerEpoch += avg_val_loss

        wandb.log({
            "checkpoint": 'OVERALL EPOCH',
            "epoch": epoch,
            "anatomy": anatomy,
            "training loss": avg_train_loss,
            "class error when training": avg_distance_train,
            "validation loss": avg_val_loss,
            "validation accuracy": avg_val_accuracy,
            "class error when testing": avg_distance_test
        })
    elapsedTime = (time.time() - script_start_time)
    print(f"Epoch {epoch}/{numOfEpochs} complete, elapsed seconds: {elapsedTime:.5f}, elapsed mins: {(elapsedTime/60):.5f} min")
    avgEpochValLoss = totalValLossPerEpoch / len(anatomies)
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avgEpochValLoss) # allows scheduler to evaluate based on val loss this epoch
    new_lr = optimizer.param_groups[0]['lr']
    if prev_lr != new_lr:
        print(f'Epoch {epoch} changed from {prev_lr:.6f} to {new_lr:.6f}')
        wandb.log({
            'Epoch' : epoch,
            "Learning Rate": new_lr
        })

wandb.finish()

print('Done with All Training and Testing')

# extract embeddings for plotting embeddings UMAP (sample code from gemini)

plottingVisualEmbeddings = True # variable to control if plotting embeddings

import json
import torch
import torch.nn.functional as F

def export_embeddings_to_json(network, all_val_sets, encoderName):
    for anatomy, val_loader in all_val_sets.items():
      network.eval()

      embeddings_dict = {}
      ssim_labels_dict = {}
      artifact_labels_dict = {}

      global_id = 0 # Acts as our "file name"

      print("Extracting embeddings to save to JSON...")

      with torch.no_grad():
          for info in val_loader:
              condition, target, prompt, target_type, anatomy, artifact, ssimCalc, validSSIM = info

              numOfSlices = condition.shape[4] # 16

              # 1. Filter and Reshape (Vectorized method)
              mask = (validSSIM == True)
              validCondition = condition[mask]
              if validCondition.shape[0] == 0: continue

              B = validCondition.shape[0]
              S = validCondition.shape[4]

              x = validCondition.permute(0, 4, 1, 2, 3).reshape(B * S, 1, 96, 96).float()
              x_min = x.view(B * S, -1).min(dim=1)[0].view(B * S, 1, 1, 1)
              x_max = x.view(B * S, -1).max(dim=1)[0].view(B * S, 1, 1, 1)
              x = (x - x_min) / (x_max - x_min + 1e-8)
              x = x.repeat(1, 3, 1, 1)
              x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

              device = next(network.parameters()).device
              x = x.to(device)
              mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
              std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
              x = (x - mean) / std

              # 2. Get 512-D Embeddings
              embeddings = network(x)
              embeddings_np = embeddings.cpu().numpy()

              # 3. Get Labels (Expanded to match all the slices)
              floatSSIM = getValidFloatSSIM(ssimCalc, validSSIM)
              ssim_classes = findTargetIndices(floatSSIM).repeat_interleave(numOfSlices).cpu().numpy()

              valid_artifacts = []

              for idx in range(len(artifact)):
                  if validSSIM[idx].item() == True:
                      valid_artifacts.append(artifact[idx])

              artifact_list = []
              for a in valid_artifacts:
                  artifact_list.extend([str(a)] * numOfSlices)

              # 4. Package into Dictionaries
              for i in range(len(embeddings_np)):
                  key = f"scan_{global_id:05d}" # e.g., "scan_00001"

                  # Wrapping in an extra list to match your original JSON format
                  embeddings_dict[key] = [embeddings_np[i].tolist()]
                  ssim_labels_dict[key] = str(ssim_classes[i])
                  artifact_labels_dict[key] = artifact_list[i]

                  global_id += 1

      # 5. Save to Files
      with open(f"{encoderName}_{anatomy}_all_embeddings.json", "w") as f:
          json.dump(embeddings_dict, f)

      with open(f"{encoderName}_{anatomy}_ssim_labels.json", "w") as f:
          json.dump(ssim_labels_dict, f)

      with open(f"{encoderName}_{anatomy}_artifact_labels.json", "w") as f:
          json.dump(artifact_labels_dict, f)

      print(f"Successfully saved {global_id} items for anatomy {anatomy} to JSON!")
      print(f'Done with encoder {encoderName}, anatomy {anatomy}, learning rate {learning_rate}, optimizer {optimizerIndex}, lambda multiplier {multiplier}, angular margin {angularMargin}')

# Run it by passing your trained network, validation set, and encoder name
export_embeddings_to_json(network, all_val_sets, encoderName)
