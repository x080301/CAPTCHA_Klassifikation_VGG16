This Project is for course "machine learning 1, basis".

CAPTCHA is short for Completely Automated Public Turing Test to Tell Computers and Humans Apart. In this project, a machine learning program should be implemented to get higher accuracy to win the bonus points for the final exam.

A VGG-19_bn network is used in thies project. It is pretrained on the ImageNet-Dataset. The parameters in the feature block of the network is fixed and setted as ''requires no grad''. The classifier block is reconstructed to match the task. The network is fine tuned on the in course given dataset.

The model reaches a accuracy of 80.4% on the given test dataset.

Besides the VGG-19_bn, a ResNet is also implemented and trained, but reaches lower accuracy.