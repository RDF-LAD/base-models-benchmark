The goal of this personal project is to get familiar with the architecture of the classical AlexNet model and the Pytorch library.

I used as trainning/test data the Cifar10 dataset that classifies 10 categories (animals,vehicles) of low quality (32*32) pictures.
I also test the model on higher quality pictures comming from internet.
The architecture, the loss, the regularization techniques and the optimizer are those of the original AlexNet paper.

The architecture is described in the CNN_archi.py file.
The Data preprocessing is done in data_importation.py.
The training is done in the trainnig.py.
The performance metrics computation is done in test.py.
The model_comparaison.py file is made to compare two models (allowing to import and compare trained model).
The POC.py file allow to test the inference of the model on any saved pictures.

The results of the models are in the results/cifar10 folder. The better overall results are reached for the model trained during 50 epochs without the regularization involving PCA on the RGB channels.
The saved models are in the models folder.