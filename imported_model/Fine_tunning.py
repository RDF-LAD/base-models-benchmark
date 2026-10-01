from Imported_archi_ViT import PreTrainedViT
from data_importation_HF import get_data_loaders
from pathlib import Path

def with_backbone_freezing(t_ds,v_ds,n_ep_1=1,n_ep_2=1,save_dir='./FT_1-1'):
    model = PreTrainedViT(10)
    model.fine_tunning(t_ds.dataset,v_ds.dataset,n_epoch=n_ep_1,b_size=32,save_dir='./Transi',backbone_training=False)
    model=PreTrainedViT(10,model_path='./Transi')
    model.fine_tunning(t_ds.dataset,v_ds.dataset,n_epoch=n_ep_2,b_size=32,save_dir=save_dir,backbone_training=True)
    

def without_backbone_freezing(t_ds,v_ds,n_ep=3,save_dir='./FT_3'):
    model = PreTrainedViT(10)
    model.fine_tunning(t_ds.dataset,v_ds.dataset,n_ep,b_size=32,save_dir=save_dir,backbone_training=False)


if __name__ == '__main__' :
    train_ds,val_ds,_ = get_data_loaders()

    with_backbone_freezing(train_ds,val_ds,n_ep_1=1,n_ep_2=1)
    

    
    
    