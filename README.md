# Welcome to the COVID-19 X-Ray Classifier

This is a simple tool written in F# using .Net Core 5.0.

## About
This is a program to train and test a DNN for image classification using the transfer learning method. This program could be customized to utlize any of the supported pretrained model, such as InceptionV3 or ResNetV2, with tiny changes. his program trains and evaluates the selected model, to identify where the patient is classified according to the following categories:

 - Typically healthy (NORMAL)
 - Confirmed COVID-19 cases (COVID-19)
 - Other viral pneumonia cases (PNEUMONIA)

This tool is dependent on the following dataset:

 - [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

# Build Instructions

> **Note:** You will require an Internet connection to build the program and run it. Web proxies are not supported, a direct Internet connection is required.

Steps:

 1. Download and install the .Net Core SDK v5.0 from [here.](https://dotnet.microsoft.com/download/dotnet/5.0)
 2. Clone this Github repository or download it using **Clone or download > Download Zip** and extract it to your hard drive.
 3. Using the command line, switch to the cloned repository or extracted folder and run the following command:`dotnet restore`
 4. Download the dataset from the above link and extract into the same folder.
 4. After a successful restore then run the following command: `dotnet build`
 5. Then finally run the program using: `dotnet run`
 6. Follow the instructions to train and test the model, like this:`dotnet run 2 ".\COVID-19 Radiography Database"`

### Authors

 - Sritharan Sivaguru

