import torch
import torch.nn as nn
import yaml

class GRUModel(nn.Module):
    def __init__(self, config_path):
        super(GRUModel, self).__init__()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.input_size = len(self.config["model"]["features"])
        self.hidden_size = self.config["training"]["hidden_size"]
        self.output_size = len(self.config["model"]["labels"])
        
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
    
    def save_model(self, path):
        """모델을 저장합니다."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, config_path, model_path):
        """저장된 모델을 불러옵니다."""
        model = cls(config_path)
        model.load_state_dict(torch.load(model_path))
        return model