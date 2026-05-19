import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn.utils import weight_norm


""" model load """
    
def get_model(model_name, num_classes, hidden_size=None, num_layers=None, dropout=None, freeze_initial_layers=None, pt=True):
    model_dict = {
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'densenet161': models.densenet161,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'mobilenet': models.mobilenet_v2,
        'efficientnet': models.efficientnet_b0,
        'googlenet': models.googlenet,
        'googlenet_aux': models.googlenet, 
        'inception': models.inception_v3,
        'cnn-gru': None,
        'res-gru': None, 
        'cnn-lstm': None,
        'res-lstm': None, 
        'convnext_t': models.convnext_tiny, 
        'convnext_s': models.convnext_small,
        'resnext': models.resnext50_32x4d
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}")
        
    elif model_name == 'cnn-gru':
        model = CNNRNN(
            num_classes=num_classes, 
            hidden_size=128, 
            num_layers=2, 
            dropout=0.5
        )
        img_size = 224
        
    elif model_name == 'res-gru':
        model = ResNetRNN(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        )
        img_size = 224
        
    elif model_name == 'cnn-lstm':
        model = CNNLSTM(
            num_classes=num_classes, 
            hidden_size=128, 
            num_layers=2, 
            dropout=0.5
        )
        img_size = 224
        
    elif model_name == 'res-lstm':
        model = ResNetLSTM(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        )
        img_size = 224
        
    elif model_name == 'googlenet_aux':
        model = model_dict[model_name](pretrained=pt, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
        img_size = 224
        
    elif model_name == 'inception':
        model = model_dict[model_name](pretrained=pt)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        img_size = 299
    
    else:
        model = model_dict[model_name](pretrained=pt)
        img_size = 224
        
        if 'resnet' in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif 'densenet' in model_name:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif 'mobilenet' in model_name:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'efficientnet' in model_name:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)      
        elif 'googlenet' in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'convnext_t' in model_name:
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)   
        elif 'convnext_s' in model_name:
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)   
        elif 'resnext' in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)  
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
        
    return model, img_size