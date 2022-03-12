# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import collections
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

scaler = MinMaxScaler(feature_range=(-1, 1))

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

df = scipy.io.loadmat('data/EVENTS.mat')
df = df['EVENTS']

# Separate majority and minority classes
df_majority0 = df[df[:, -1] != 1 ]
df_majority1 = df[df[:, -1] == 1]
# df_majority3 = df[df[:, -1] == 3]

# df_minority2 = df[df[:, -1] == 2]  # 550

# Downsample majority class
df_majority0_downsampled = resample(df_majority0,
                                    replace=False,  # sample without replacement
                                    n_samples=792,  # to match minority class
                                    random_state=123)  # reproducible results
#
# df_majority1_downsampled = resample(df_majority1,
#                                    replace=False,  # sample without replacement
#                                    n_samples=df_majority1,  # to match minority class
#                                    random_state=123)  # reproducible results
#
# df_majority3_downsampled = resample(df_majority3,
#                                    replace=False,  # sample without replacement
#                                    n_samples=df_majority1,  # to match minority class
#                                    random_state=123)  # reproducible results

df_downsampled = np.concatenate([df_majority0_downsampled, df_majority1])
np.random.shuffle(df_downsampled)

features_numpy = df_downsampled[:, :-1]
targets_numpy = df_downsampled[:, -1]
count = collections.Counter(targets_numpy)
targets_numpy[targets_numpy != 1] = 0
#
# # min_max_scaler = preprocessing.MinMaxScaler()
# # features_numpy = min_max_scaler.fit_transform(features_numpy)
#
# # features_numpy = scaler.fit_transform(features_numpy)
#
# # features_numpy = preprocessing.scale(features_numpy)
#
mean = features_numpy.mean(axis=0)
features_numpy -= mean
std = features_numpy.std(axis=0)
features_numpy /= std

# train test split. Size of train data is 80% and size of test data is 20%.
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                              targets_numpy,
                                                                              test_size=0.2,
                                                                              random_state=42)

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients.
# Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)  # .type(torch.double)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)  # data type is long
# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)  # .type(torch.double)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest)

# data loader
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


# visualize one of the images in data set
# plt.plot(features_numpy[10,:])
# plt.axis("off")
# plt.title(str(targets_numpy[10]))
# plt.show()

# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, p):
        super(RNNModel, self).__init__()
        self.p = p
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn1 = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.rnn2 = nn.RNN(hidden_dim, 20, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(20, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        h1 = Variable(torch.zeros(self.layer_dim, x.size(0), 20)).to(device)

        # One time step
        out, hn = self.rnn1(x, h0)
        out = nn.functional.dropout(out, p=self.p, training=True)
        out, hn2 = self.rnn2(out, h1)
        out = nn.functional.dropout(out, p=self.p, training=True)
        out = self.fc(out[:, -1, :])
        return out


# Create RNN
input_dim = 1250  # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 1  # number of hidden layers
output_dim = 2  # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, 0.3)
model = model.float()
model.to(device)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 1
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
for epoch in range(num_epochs):
    totall = 0
    correctt = 0
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels).to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train.float().to(device))

        predicted = torch.max(outputs.data, 1)[1]
        totall += labels.size(0)
        correctt += (predicted.to(device) == labels).sum()

        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1
        # print(count)
        if count % 10 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim))

                # Forward propagation
                outputs = model(images.float().to(device))

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += labels.size(0)

                correct += (predicted.to(device) == labels.to(device)).sum()

            accuracy = 100 * correct / float(total)


            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 100 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

    print(100 * correctt / float(totall))

print("The state dict keys: \n\n", model.state_dict().keys())
checkpoint = {'model': RNNModel(input_dim, hidden_dim, layer_dim, output_dim, p=0.3),
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

model = load_checkpoint('checkpoint.pth')
print(model)
model.to(device)

df_test = scipy.io.loadmat('data/subj12.mat')
df_test = df_test['subj12']
df_test = df_test[0,0]

predict = np.array([]).reshape(0, df_test[0].shape[0])
for j in range(12):
    df_test_ = df_test[j]
    mean = df_test_.mean(axis=0)
    df_test_ -= mean
    std = df_test_.std(axis=0)
    df_test_ /= std
    df_test_ = torch.from_numpy(df_test_)  # .type(torch.double)

    Test = Variable(df_test_.view(-1, seq_dim, input_dim))
    outputs = model(Test.float().to(device))
    predicted = torch.max(outputs.data, 1)[1]
    predicted = predicted.cpu().numpy()
    print(collections.Counter(predicted))

    predict = np.concatenate([predict, predicted.reshape(1, -1)])
#
#
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
#
# # tp, fp, tn, fn = confusion(torch.from_numpy(predicted), torch.from_numpy(targets_numpy))
#
#
df_test_ = torch.from_numpy(features_test)  # .type(torch.double)

Test = Variable(df_test_.view(-1, seq_dim, input_dim))
outputs = model(Test.float().to(device))
predicted = torch.max(outputs.data, 1)[1]
predicted = predicted.cpu().numpy()
#
# from sklearn.metrics import roc_curve, auc
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(predicted, targets_test)
# AUC = auc(fpr_keras, tpr_keras)
# plt.plot(fpr_keras, tpr_keras, label='RNN Model(area = {:.3f})'.format(AUC))
# plt.xlabel('False positive Rate')
# plt.ylabel('True positive Rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
#
# pred = pd.DataFrame(predict)
# pred.to_csv('data/predict12.csv', header=False, index=False)