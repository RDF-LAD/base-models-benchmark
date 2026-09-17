import torch
import torch.nn as nn
from torchinfo import summary

class PatchEmbedding(nn.Module):
    """Convertit une image en une séquence de patches et applique une projection linéaire."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Utilisation d'une convolution 2D pour découper et projeter efficacement les patches
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x shape: [Batch_Size, Channels, Height, Width]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """Architecture Vision Transformer (ViT) complète."""
    def __init__(
        self, 
        img_size=32, 
        patch_size=4, 
        in_channels=3, 
        num_classes=1000, 
        embed_dim=384, depth=6, num_heads=6, mlp_ratio=2 ,
        drop_rate=0.0
    ):
        super().__init__()
        
        # 1. Découpage en patches et projection
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 2. Tokens et embeddings positionnels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 3. Blocs du Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),# Qu'est ce que le MLP ratio ? -> C'est le facteur par lequel les dimensions du premier layer augmente l'input avant de le reduire symetriquement
            dropout=drop_rate,
            activation='gelu', # Qu'est ce que la fonction d'activation gelu ? fonction qui pondère les entrées en fonction de leur valeur (non linéarité via tanh)
            batch_first=True,
            norm_first=True # Norme avant l'attention (Pre-LN, standard pour ViT)
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Normalisation finale et tête de classification
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialisation des poids
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # Extraction des patches -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        
        # Ajout du token de classification [CLS] au début de la séquence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Ajout des embeddings positionnels et application du dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Passage à travers les blocs Transformer
        x = self.blocks(x)
        
        # Normalisation finale
        x = self.norm(x)
        
        # Récupération du token [CLS] pour la classification
        x = x[:, 0]
        
        # Tête de classification linéaire
        x = self.head(x)
        return x

# Exemple d'instanciation et de test rapide
if __name__ == "__main__":
    model = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=10, 
        embed_dim=768, 
        depth=12, 
        num_heads=12
    )
    
    # Image factice [Batch_Size, Channels, Height, Width]
    dummy_image = torch.randn(2, 3, 224, 224)
    output = model(dummy_image)
    print("Dimensions de la sortie :", output.shape)  # Attendu : torch.Size([2, 10])

# example_model = SimpleResNet(num_classes=10)
# summary(example_model,input_size=(1, 3, 224, 224))