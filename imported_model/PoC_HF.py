from Imported_archi_ViT import PreTrainedViT


def infer(model_path,img_path):

    model = PreTrainedViT(num_classes=10,model_path=model_path)
    model.inference(img_path)

if __name__ == '__main__':
    m_path= r'C:\Users\User\Documents\projets vision\AlexNet\models\FT_3'
    im_path = r'C:\Users\User\Documents\projets vision\AlexNet\test_pictures\avion1.jpg'
    infer(m_path,im_path)