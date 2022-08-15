# MTCSNN: Multi-task Clinical Siamese Neural Network for Diabetic Retinopathy Severity Prediction
This is the course group project of UMich EECS 545 (2022 Winter).

Diabetic Retinopathy (DR) has become one of the leading causes of vision impairment in working-aged people and is a severecoproblem worldwide. However, most of the works ignored the ordinal information of labels. In this project, we propose a novel design MTCSNN, a Multi-task Clinical Siamese Neural Network for Diabetic Retinopathy severity prediction task. The novelty of this project is to utilize the ordinal information among labels and add a new regression task, which can help the model learn more discriminative feature embedding for fine-grained classification tasks. We perform comprehensive experiments over the RetinaMNIST, comparing MTCSNN with other models like ResNet-18, 34, 50. Our results indicate that MTCSNN out-performs the benchmark models in terms of AUC and accuracy on the test dataset. 

## Code Structure

- Model architecture implementation based on the code provided by [torchvision.models.resnet](https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html)：
  - `resnet18.py`
  - `resnet34.py`
  - `resnet50.py`
- Dataset from [MedMNIST](https://github.com/MedMNIST/MedMNIST)
  - `dataset.py`: PyTorch datasets and dataloaders of MedMNIST
  - `evaluator.py`: Standardized evaluation functions
  - `info.py`: Dataset information `dict` for each subset of MedMNIST

