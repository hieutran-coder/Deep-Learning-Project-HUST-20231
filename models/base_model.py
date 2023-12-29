import torch
import torch.nn as nn
import timm

from typing import List


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(0, '')
        self.model.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x
    

class Classifier(nn.Module):
    def __init__(self, 
                 in_features: int, num_classes: int, 
                 mlp_structures: List[int] = [2, ], 
                 drop_rate: float = 0.2
                ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.mlp_structures = mlp_structures

        self.head = nn.Linear(self.in_features, self.num_classes)
        
        layers = []
        for i in range(len(self.mlp_structures) + 1):
            if i == 0:
                layers.append(nn.Linear(self.in_features, self.in_features * self.mlp_structures[0]))
            elif i == len(self.mlp_structures):
                layers.append(nn.Linear(self.in_features * self.mlp_structures[-1], self.in_features))
            else:
                layers.append(nn.Linear(self.in_features * self.mlp_structures[i - 1], self.in_features * self.mlp_structures[i]))
            
            if i == len(self.mlp_structures) // 2 and i != len(self.mlp_structures) and i != 0:
                layers.append(nn.Dropout(drop_rate))
                layers.append(nn.LayerNorm(self.in_features * self.mlp_structures[i]))
            elif i == len(self.mlp_structures):
                layers.append(nn.Dropout(drop_rate))
            else:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(drop_rate))

        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        out = self.mlp(x)
        out += x
        out = self.head(out)
        return out
    


class Model(nn.Module):
    def __init__(self, model_name: str, num_classes: int, mlp_structures: List[int] = [2, ], drop_rate: float = 0.2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(self.model_name, True)
        self.classifier = Classifier(self.feature_extractor.model.num_features, self.num_classes, mlp_structures=mlp_structures, drop_rate=drop_rate)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x