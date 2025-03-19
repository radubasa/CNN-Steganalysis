# CNN-Steganalysis

## Overview

**StegaScanMail** is a deep learning Python-based tool designed for steganalysis on images in received emails. It utilizes a convolutional neural network (CNN) to classify both greyscale and RGB images, specifically targeting images encoded with the least significant bit (LSB) technique. This tool is particularly useful for detecting malicious code embedded in images, a common tactic used by cyber attackers due to its simplicity and effectiveness in avoiding detection.

The training and testing of StegaScanMail were performed using the **Stego-Images-Dataset**, which contains 44,000 PNG images embedded with malicious code. The PNG format is frequently used in emails for logos and web graphics, making it a prime target for cyberattacks.

## Features

- **Deep Learning-Based Steganalysis**: Utilizes a CNN to detect steganographic content in images.
- **Supports Greyscale and RGB Images**: Capable of analyzing both types of images.
- **Focus on PNG Format**: Trained primarily on PNG images encoded with the LSB technique.
- **Cloud Deployment**: Can be deployed as a Docker container in the cloud.
- **Open Source**: Developed as part of a master's thesis and made available for public use.

## Author

This project was developed by Radu-Cristian Basarabă as part of a master's thesis.

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker
- Azure CLI (for cloud deployment)

### Clone the Repository

```bash
git clone https://github.com/yourusername/CNN-Steganalysis.git
cd CNN-Steganalysis

```

## Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  
pip install -r [requirements.txt]
```

## Training the Model

To train the model, run the following command:

```bash
python [ModelTraining+metrics.py]
```

## Running the Application

To run the application locally:

```bash
python [metricsCNN.py]
```

## Deployment

### Docker Deployment

#### Build the Docker image:

```bash
docker build -t stegascanner .
```

#### Run the Docker container:

```bash
docker run -d -p 5000:5000 stegascanner
```

### Azure Cloud Deployment

1. **Login to Azure**

    ```bash
    az login
    ```

2. **Create a Resource Group**

    ```bash
    az group create --name StegaScanMailGroup --location eastus
    ```

3. **Create an Azure Container Registry**

    ```bash
    az acr create --resource-group StegaScanMailGroup --name StegaScanMailRegistry --sku Basic
    ```

4. **Build and Push Docker Image to ACR**

    ```bash
    az acr build --registry StegaScanMailRegistry --image stegascanner .
    ```

5. **Create an Azure Container Instance**

    ```bash
    az container create --resource-group StegaScanMailGroup --name StegaScanMailInstance --image StegaScanMa
    ```

6. **Get the Public IP Address**

    ```bash
    az container show --resource-group StegaScanMailGroup --name StegaScanMailInstance --query ipAddress.ip
    ```

Access the application at the provided IP address.

## Project Structure


```markdown
CNN-Steganalysis
├── src
│   ├── Dataset.py
│   ├── EmailProcessor.py
│   ├── Fine-tuned CNN.py
│   ├── validate_model.py
│   ├── ImagePreprocessor.py
│   ├── metricsClass.py
│   ├── ResNet50.py
│   ├── Statistical Methods + SVM.py
│   ├── SteganalysisModelLoad.py
│   └── VGG166.py
├── data
│   ├── stego_images_dataset
│   │   ├── train
│   │   ├── validation
│   │   └── test
│   ├── bossbase (optional)
│   └── alaska (optional)
├── models
│   └── best_modelCNN.keras
├── results
│   ├── Model1.txt
│   ├── Model2.txt
│   └── Model3.txt
├── [requirements.txt]
└── [README.md]
```


## How to Download and Deploy

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CNN-Steganalysis.git
cd CNN-Steganalysis
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  
pip install -r [requirements.txt]
```

### 3. Run the Application

To run the application locally:

```bash
python [metricsCNN.py]
```

### 4. Deploy to Azure

Follow the steps in the [Azure Cloud Deployment](#azure-cloud-deployment) section above.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or inquiries, please contact Radu-Cristian Basarabă at [r.basaraba@student.utwente.nl].

This project is made open source for educational purposes and as part of a master's thesis.
```