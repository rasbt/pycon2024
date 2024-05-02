# %% [markdown]
# # Logistic Regression Classifier

# %% [markdown]
# ## 1) Installing Libraries

# %%
%load_ext watermark
%watermark -v -p numpy,pandas,matplotlib,torch -conda

# %% [markdown]
# ## 2) Loading the Dataset

# %%
import pandas as pd

df = pd.read_csv("toydata-truncated.txt", sep="\t")
df

# %%
X_train = df[["x1", "x2"]].values
y_train = df["label"].values

# %%
X_train

# %%
X_train.shape

# %%
y_train

# %%
y_train.shape

# %%
import numpy as np

np.bincount(y_train)

# %% [markdown]
# ## 3) Visualizing the dataset

# %%
%matplotlib inline
import matplotlib.pyplot as plt

# %%
plt.plot(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

plt.plot(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()

# %%
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

# %%
plt.plot(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

plt.plot(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()

# %% [markdown]
# ## 4) Implementing the model

# %%
import torch
import torch.nn.functional as F

class LogisticRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits

# %%
torch.manual_seed(1)

model = LogisticRegression(num_features=2, num_classes=2)

# %%
x = torch.tensor([[1.1, 2.1],
                  [1.1, 2.1],
                  [9.1, 4.1]])

with torch.no_grad():
    logits = model(x)
    probas = F.softmax(logits, dim=1)

print(probas)

# %% [markdown]
# ## 5) Defining a DataLoader

# %%
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):

        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.labels.shape[0]


train_ds = MyDataset(X_train, y_train)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=10,
    shuffle=True,
)

# %%
X_train.shape

# %% [markdown]
# ## 6) The training loop

# %%
import torch.nn.functional as F


torch.manual_seed(1)
model = LogisticRegression(num_features=2, num_classes=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 20

for epoch in range(num_epochs):

    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):


        ###############################
        ### Complete the training loop
        ###
        ### Your code below
        ###############################

        logits = # ?????

        loss = F.cross_entropy(logits, class_labels)

        # ?????
        # ?????
        # ?????

        ################################
        ## No changes necessary below
        ################################

        ### LOGGING
        print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}'
               f' | Batch {batch_idx:03d}/{len(train_loader):03d}'
               f' | Loss: {loss:.2f}')


# %% [markdown]
# ## 7) Evaluating the results

# %%
def compute_accuracy(model, dataloader):

    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, class_labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        pred = torch.argmax(logits, dim=1)

        compare = class_labels == pred
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples

# %%
train_acc = compute_accuracy(model, train_loader)

# %%
print(f"Accuracy: {train_acc*100}%")

# %% [markdown]
# ## 8) Optional: visualizing the decision boundary

# %%
plt.plot(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

plt.plot(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()

# %%
def plot_boundary(model):

    w1 = model.linear.weight[0][0].detach()
    w2 = model.linear.weight[0][1].detach()
    b = model.linear.bias[0].detach()

    x1_min = -20
    x2_min = (-(w1 * x1_min) - b) / w2

    x1_max = 20
    x2_max = (-(w1 * x1_max) - b) / w2

    return x1_min, x1_max, x2_min, x2_max

# %%
x1_min, x1_max, x2_min, x2_max = plot_boundary(model)


plt.plot(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

plt.plot(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k")

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()


