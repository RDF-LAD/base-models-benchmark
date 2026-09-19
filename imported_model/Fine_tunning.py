from Imported_archi_ViT import PreTrainedViT
from data_importation_HF import get_data_loaders

if __name__ == '__main__' :
    model = PreTrainedViT(10)

    train_ds,val_ds,_ = get_data_loaders()
    model.fine_tunning(train_ds.dataset,val_ds.dataset,n_epoch=3,b_size=32,save_dir='./FT_3',backbone_training=False)
    