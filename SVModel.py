import torch.nn as nn
import torchvision.models as models

class SVRegressor(nn.Module):
    def __init__(self, arch_name, in_channel=3, pre_trained=False, num_shoe_params=22):
        super(SVRegressor, self).__init__()
        self.arch_name = arch_name
        self.in_channel = in_channel
        self.pre_trained = pre_trained
        self.num_shoe_params = num_shoe_params
        self.build_model()
        
        
    def build_model(self):
        print(f'Building {self.arch_name} model (pretrained={self.pre_trained})!!')
        
        if self.arch_name == 'vgg16':
            base_model = models.vgg16(pretrained=self.pre_trained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.regressor = base_model.classifier
            
            #Replace the classification head with regression head
            self.regressor[6] =  nn.Sequential(
                nn.Linear(self.regressor[3].out_features, self.num_shoe_params),
                nn.ReLU())
            
            if self.pre_trained:
                #Freeze all the weights before CB5
                for param in self.features[0:24].parameters():
                    param.requires_grad = False
            
                
        elif self.arch_name == 'resnet50':
            base_model = models.resnet50(pretrained=self.pre_trained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            if self.in_channel == 1:
                self.features[0] =  nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.regressor = nn.Sequential(
                nn.Linear(base_model.fc.in_features, self.num_shoe_params),
                nn.ReLU())
            
            if self.pre_trained:
                i = 4 if self.in_channel == 1 else 0
                # Freeze all weights before CB4
                for param in self.features[i:6].parameters():
                    param.requires_grad = False
        elif self.arch_name == 'resnet18':
            base_model = models.resnet18(pretrained=self.pre_trained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.regressor = nn.Sequential(
                nn.Linear(base_model.fc.in_features, self.num_shoe_params),
                nn.Sigmoid()
            )
            if self.pre_trained:
                for param in self.features[0:6].parameters():
                    param.requires_grad = False      
                
        else:
            raise('This architecture is not supported!!')
                                          

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.shape[0], -1)
        y = self.regressor(feat)
        return y
        
if __name__ == '__main__':
    print('Okay')