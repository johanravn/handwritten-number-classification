
Running single model:
Download wheights from https://drive.google.com/open?id=1iPih4XqFdaX6a1Ue10G-dOhprG0Dhoi5 and extract them into the weights folder

Uses a single resnet50 model for f1-score should be 0.970.

python3 resnet50_test_single_digit.py

Enesembled model:

Consist of 3 models trained from ResNet50 architecture, and 4 models trained from InceptionResNetV2.

f1-score should be 0.975

python3 resnet50_test_ensemble_single_digit.py