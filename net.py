from torch import nn

def get_model(feature_size, result_num, dp1=0.2, dp2=0.5):
    net = nn.Sequential(
                    nn.Linear(feature_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dp1),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(dp2),
                    nn.Linear(128, result_num))
    return net