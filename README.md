# plant-disease-classifier
A resnet50 fine tuned on plantvillage dataset for classification of plants and diseases it has from an image of its leaf.

## Dataset
I have used BrandonFors/Plant-Diseases-PlantVillage-Dataset from hugging face for this model. It contains 43.5K observatiojs with 38 classes

## Model
I have used a pretrained resnet50 and fine tuned its last layer on the dataset

## Preprocessing
I have used random resized crop and random horizontal flip for data augmentation during fine tuning and a normalize with resize transforms for evaluation.