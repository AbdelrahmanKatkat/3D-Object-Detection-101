# Projects

- VoxelNetXt KITTI
    
    The KITTI dataset is a popular computer vision dataset that was created for the development and evaluation of various computer vision tasks, such as object detection, tracking, and 3D reconstruction. It was collected and released by the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago.
    
    Some key points about the KITTI dataset:
    
    1. Autonomous Driving Scenes: The dataset contains a wide variety of scenes and scenarios captured from a vehicle driving around a mid-sized city, including urban, rural, and highway environments.
    2. Sensor Data: The dataset includes synchronized data from various sensors, including high-resolution color and grayscale cameras, a velodyne 3D laser scanner, and a GPS/IMU system, providing rich information for various computer vision tasks.
    3. Annotated Objects: The dataset provides detailed annotations for various objects of interest, such as cars, pedestrians, and cyclists, including their 2D bounding boxes, 3D bounding boxes, and object attributes.
    4. Evaluation Benchmarks: The KITTI dataset has been widely adopted as a benchmark for evaluating the performance of computer vision algorithms in tasks such as object detection, tracking, and 3D object detection.
    5. Variety of Challenges: The dataset presents various challenges, such as varying illumination conditions, occlusions, and the presence of a wide range of object sizes, making it a valuable resource for developing robust and generalizable computer vision algorithms.
    
    The KITTI dataset has been extensively used in the computer vision research community and has contributed to the development of many state-of-the-art algorithms and techniques in autonomous driving and related domains.
    
    [LiDAR point-cloud based 3D object detection implementation with colab {Part-1 of 2}](https://medium.com/towards-data-science/lidar-point-cloud-based-3d-object-detection-implementation-with-colab-part-1-of-2-e3999ea8fdd4)
    
    [LiDAR point cloud based 3D object detection implementation with colab{Part 2 of 2}](https://towardsdatascience.com/lidar-point-cloud-based-3d-object-detection-implementation-with-colab-part-2-of-2-f3ad55c3f38c)
    
    [GitHub - dvlab-research/VoxelNeXt: VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking (CVPR 2023)](https://github.com/dvlab-research/VoxelNeXt?tab=readme-ov-file)
    
- PointNet Object Detection
    
    [https://www.digitalnuage.com/pointnet-or-the-first-neural-network-to-handle-directly-3d-point-clouds](https://www.digitalnuage.com/pointnet-or-the-first-neural-network-to-handle-directly-3d-point-clouds)
    
- **3D ResNet-18 Project**
    
    **1. Data Collection and Preprocessing:**
    Collect a dataset of 3D dental scans containing instances of caries lesions. Preprocess the data to ensure uniformity and quality, including voxelization, normalization, and augmentation techniques such as rotation, translation, and scaling to increase dataset diversity.
    
    ```python
    # Example preprocessing steps using PyTorch
    import torch
    from torchvision.transforms import Compose, Resize, Normalize
    from torch.utils.data import DataLoader
    from custom_dataset import CustomDataset
    
    # Define data transformations
    transform = Compose([
        Resize((224, 224)),  # Resize input to fixed size
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize input
    ])
    
    # Create dataset and dataloader
    dataset = CustomDataset(data_dir='path_to_data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    ```
    
    **2. Model Architecture Selection:**
    Choose a suitable architecture for 3D object detection. Common choices include 3D variants of popular CNN architectures like ResNet, VGG, or custom architectures designed specifically for 3D data processing.
    
    ```python
    import torch.nn as nn
    import torchvision.models as models
    
    # Example model architecture (3D ResNet)
    class CariesDetectionModel(nn.Module):
        def __init__(self, num_classes):
            super(CariesDetectionModel, self).__init__()
            self.base_model = models.video.r3d_18(pretrained=True)
            self.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
        def forward(self, x):
            x = self.base_model(x)
            x = self.fc(x)
            return x
    
    # Instantiate the model
    model = CariesDetectionModel(num_classes=2)  # Assuming binary classification (caries vs. non-caries)
    
    ```
    
    **3. Loss Function Definition:**
    Define a suitable loss function to optimize the model during training. For object detection tasks, common choices include cross-entropy loss or focal loss, combined with regression losses like smooth L1 loss for bounding box regression tasks.
    
    ```python
    import torch.optim as optim
    
    # Example loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    ```
    
    **4. Training:**
    Train the model using the prepared dataset and defined loss function. Iterate through the dataset in mini-batches, compute the loss, and update the model parameters using backpropagation.
    
    ```python
    # Example training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['image'], data['label']
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    
    print('Finished Training')
    
    ```
    
    **5. Evaluation:**
    Evaluate the trained model on a separate validation set to assess its performance. Compute metrics such as accuracy, precision, recall, and F1 score to evaluate the model's effectiveness in detecting caries lesions.
    
    ```python
    # Example evaluation loop
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['image'], data['label']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    ```
    
    **6. Fine-tuning and Optimization:**
    Fine-tune the model and optimize hyperparameters based on validation performance. Experiment with different learning rates, batch sizes, and augmentation strategies to improve detection accuracy and robustness.
    
    ```python
    # Example fine-tuning and optimization
    # Adjust learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Train for additional epochs
    for epoch in range(5):
        scheduler.step()
        # Training loop...
    
    ```
    
    **7. Deployment and Inference:**
    Deploy the trained model for inference on new 3D dental scans to detect caries lesions. Process input data, feed it through the trained model, and interpret the output predictions.
    
    ```python
    # Example inference
    def predict(image):
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        return predicted
    
    # Load a new 3D dental scan
    new_scan = load_scan('path_to_scan')
    # Preprocess the scan
    processed_scan = preprocess_scan(new_scan)
    # Perform inference
    prediction = predict(processed_scan)
    print("Caries detected:", prediction)
    
    ```
    
    **8. Conclusion:**
    Building a caries detection model involves several key steps, including data preparation, model selection, training, evaluation, and deployment. By following these steps and leveraging the capabilities of PyTorch, one can develop an effective 3D object detection model for caries lesions in dental scans.
    
- **3D PointNet Project**
    
    Let's use an architecture suitable for 3D object detection. One popular choice is the PointNet architecture, which directly operates on point clouds without requiring voxelization. Here's how you can implement the main steps using PointNet architecture in PyTorch:
    
    **1. Data Collection and Preprocessing:**
    Collect a dataset of 3D dental scans containing instances of caries lesions. Preprocess the data to ensure uniformity and quality, including normalization and augmentation techniques.
    
    ```python
    # Example preprocessing steps for PointNet
    import torch
    from torchvision.transforms import Compose, Normalize
    from torch.utils.data import DataLoader
    from custom_dataset import CustomDataset
    
    # Define data transformations
    transform = Compose([
        Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # Normalize input
    ])
    
    # Create dataset and dataloader
    dataset = CustomDataset(data_dir='path_to_data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    ```
    
    **2. Model Architecture Selection:**
    Implement the PointNet architecture for 3D object detection. PointNet processes each point individually and aggregates features using max-pooling and fully connected layers.
    
    ```python
    import torch.nn as nn
    
    # PointNet architecture
    class PointNet(nn.Module):
        def __init__(self, num_classes):
            super(PointNet, self).__init__()
            self.conv1 = nn.Conv1d(3, 64, 1)  # Input channels: 3 (x, y, z coordinates), Output channels: 64
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
    
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Instantiate the model
    model = PointNet(num_classes=2)  # Assuming binary classification (caries vs. non-caries)
    
    ```
    
    **3. Loss Function Definition:**
    Define a suitable loss function for training the PointNet model. For classification tasks, you can use cross-entropy loss.
    
    ```python
    import torch.optim as optim
    
    # Example loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    ```
    
    **4. Training:**
    Train the PointNet model using the prepared dataset and defined loss function. Iterate through the dataset in mini-batches, compute the loss, and update the model parameters using backpropagation.
    
    ```python
    # Example training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['point_cloud'], data['label']
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    
    print('Finished Training')
    
    ```
    
    **5. Evaluation:**
    Evaluate the trained PointNet model on a separate validation set to assess its performance. Compute metrics such as accuracy, precision, recall, and F1 score to evaluate the model's effectiveness in detecting caries lesions.
    
    **6. Fine-tuning and Optimization:**
    Fine-tune the PointNet model and optimize hyperparameters based on validation performance. Experiment with different learning rates, batch sizes, and augmentation strategies to improve detection accuracy and robustness.
    
    **7. Deployment and Inference:**
    Deploy the trained PointNet model for inference on new 3D dental scans to detect caries lesions. Process input data, feed it through the trained model, and interpret the output predictions.
    
    **8. Conclusion:**
    By following these steps and leveraging the PointNet architecture in PyTorch, you can build an effective 3D object detection model for caries lesions in dental scans. Adjust the model architecture and training parameters as needed to achieve optimal performance.
    
- Dentist Project
    
    [GitHub - limhoyeon/ToothGroupNetwork: 3D Dental surface segmentation with Tooth Group Network](https://github.com/limhoyeon/ToothGroupNetwork?tab=readme-ov-file)
    
    [https://github.com/abenhamadou/3DTeethSeg22_challenge](https://github.com/abenhamadou/3DTeethSeg22_challenge)