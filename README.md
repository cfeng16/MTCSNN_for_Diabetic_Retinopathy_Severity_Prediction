# MTCSNN: Multi-task Clinical Siamese Neural Network for Diabetic Retinopathy Severity Prediction

Diabetic Retinopathy (DR) has become one of the leading causes of vision impairment in working-aged people and is a severe problem worldwide. However, most of the works ignored the ordinal information of labels. In this project, we propose a novel design MTCSNN, a Multi-task Clinical Siamese Neural Network for Diabetic Retinopathy severity prediction task. The novelty of this project is to utilize the ordinal information among labels and add a new regression task, which can help the model learn more discriminative feature embedding for fine-grained classification tasks. We perform comprehensive experiments over the RetinaMNIST, comparing MTCSNN with other models like ResNet-18, 34, 50. Our results indicate that MTCSNN outperforms the benchmark models in terms of AUC and accuracy on the test dataset. 

## Model Architecture
<p align="center">
  <img src="https://github.com/cfeng16/MTCSNN_for_Diabetic_Retinopathy_Severity_Prediction/assets/71869595/f64259d1-d284-4311-bbc9-8111a10ee53e" width="75%" height="75%">
</p>
  
## Loss function
<p align="center">
  <img width="224" alt="image" src="https://github.com/cfeng16/MTCSNN_for_Diabetic_Retinopathy_Severity_Prediction/assets/71869595/dfa12874-eaf0-4dce-9705-e132a6509285">
</p>
L1 is the general cross-entropy loss employed in the classification task while L2 is the mean square error (MSE) loss targeting the difference regression task, which also acts as a form of regularization.

## Experiment Results
<p align="center">
  <img width="459" alt="image" src="https://github.com/cfeng16/MTCSNN_for_Diabetic_Retinopathy_Severity_Prediction/assets/71869595/cf7b7b2b-b1c7-4b6b-b9d1-f8dfa33e3a5a">
  <img width="536" alt="image" src="https://github.com/cfeng16/MTCSNN_for_Diabetic_Retinopathy_Severity_Prediction/assets/71869595/987489b9-8c59-406c-915c-da7a1a55b720">
  <img width="507" alt="image" src="https://github.com/cfeng16/MTCSNN_for_Diabetic_Retinopathy_Severity_Prediction/assets/71869595/732c3587-864a-477e-b440-938af06df89e">
</p>

## Code Structure

- Model architecture implementation based on the code provided by [torchvision.models.resnet](https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html)：
  - `resnet18.py`
  - `resnet34.py`
  - `resnet50.py`
- Dataset from [MedMNIST](https://github.com/MedMNIST/MedMNIST)
  - `dataset.py`: PyTorch datasets and dataloaders of MedMNIST
  - `evaluator.py`: Standardized evaluation functions
  - `info.py`: Dataset information `dict` for each subset of MedMNIST
