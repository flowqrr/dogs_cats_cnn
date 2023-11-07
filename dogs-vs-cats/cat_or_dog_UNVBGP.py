import torch

import torch.nn as nn
import streamlit as st

from torchvision import transforms
from PIL import Image

# region model

class ConvBlock(nn.Module):
    def __init__(self,
                 n_input_features,
                 n_output_features,
                 kernel_size,
                 stride,
                 padding,
                 bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n_input_features,
                      out_channels=n_output_features,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            # ensure that the inputs to each layer have a consistent scale and mean
            # more stable training & faster convergence
            nn.BatchNorm2d(n_output_features),
            nn.ReLU(inplace=True)
        )

    # applies the convolutional block to the input image
    def forward(self, input_image):
        return self.conv(input_image)


class MyModel(nn.Module):

    # input features = 3 (for the 3 color channels)
    def __init__(self, input_features=3):
        super(MyModel, self).__init__()

        self.conv1 = ConvBlock(n_input_features=input_features,
                               n_output_features=64,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.conv2 = ConvBlock(n_input_features=64,
                               n_output_features=128,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # dimension reduction
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256, 3, 1, 1)
        self.conv4 = ConvBlock(256, 512, 3, 1, 1)
        self.conv5 = ConvBlock(512, 512, 3, 1, 1)

        # dimension reduction
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(512, 300),
            # randomly sets a fraction of input units to zero during training
            # helps prevent overfitting by adding noise to the network
            nn.Dropout(0.5),
            nn.Linear(300, 300),
            nn.Dropout(0.5),
            nn.Linear(300, 1),
            # squash the output to the range [0, 1]
            # used bc of the binary classification
            nn.Sigmoid()
        )

    # passing the input through the layers
    def forward(self, input_img):
        x = self.conv1(input_img)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = MyModel()
model = torch.load("pytorch_model.pth").to('cuda')
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4884, 0.4551, 0.4170], std=[0.2256, 0.2210, 0.2214])
])

# endregion

# region testing locally

# image_path = "E:/dogs_vs_cats/train/train/dog.3.jpg"
# image = Image.open(image_path)
# image = preprocess(image).unsqueeze(0).to('cuda')
#
# with torch.no_grad():
#     output = model(image)
#     predicted = (output > 0.5).int()
#
# predicted = predicted.cpu().numpy()
# predicted_label = "cat" if predicted == 1 else "dog"

# endregion

# region streamlit

st.title("Cat or Dog Image Classifier")
# file uploader widget
uploaded_image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # displaying the image
    image = Image.open(uploaded_image)
    left_column, center_column, right_column = st.columns(3)
    with center_column:
        st.image(image, width=300)

    # preparing the image
    image = preprocess(image).unsqueeze(0).to('cuda')

    # make a prediction
    with torch.no_grad():
        output = model(image)
        predicted = (output > 0.5).int()

    predicted = predicted.cpu().numpy()
    predicted_label = "cat" if predicted == 1 else "dog"

    # display the prediction
    with center_column:
        st.write(f"This is a {predicted_label}.")

# endregion
