import torch
from data_importation_HF import get_data_loaders
from Imported_archi_ViT import PreTrainedViT
from transformers import AutoModelForImageClassification

def test(model_path = None, n_class = 10):
    if model_path == None:
        model = PreTrainedViT(n_class)
    else :
        model = PreTrainedViT(n_class,model_path=model_path)

    _,_,test_ds = get_data_loaders()

    model.test(test_ds)
    return('End_test')

if __name__ == '__main__' :
    test()
        