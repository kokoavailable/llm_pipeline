import torch
import torch.nn as nn
import yaml

class GRUModel(nn.Module):
    def __init__(self, config_path):
        # nn.Module을 상속받아 내부구조를 초기화한다.
        # 파이토치에서 모든 모델은 모듈을 상속 받으며, 부모 클래스의 메소드를 상속한다.
        
        # Python 3 이상에서는 그냥 super().__init__() 으로 써도 무방하다.
        super(GRUModel, self).__init__()
        
        # config 파일(.yaml)을 열어 설정 로드.
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 입력데이터의 feature(속성, 특징, 컬럼) 개수를 가져온다
        self.input_size = len(self.config["model"]["features"])

        # GRU 레이어의 hidden state 차원 수. 모델의 직접적 학습 능력과 연관된다.
        self.hidden_size = self.config["training"]["hidden_size"]

        # 출력 feature 수. 예측하고자 하는 target의 수
        self.output_size = len(self.config["model"]["labels"])
        
        # GRU 레이어 정의 (gating 문을 도입하여, RNN에서 기억유지를 조절했다.)
        # 입력텐서의 차원을 배치, 시퀀스 길이, 특성수로 받는다.
        # 각 시점에 대해 GRU가 뽑은 은닉상태 (정보 요약 벡터)가 나온다.
        # batch 사이즈와 시퀀스 길이는 동적 배치 자원.
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        # 완전 연결 선형 레이어의 정의
        # 입력 벡터를 출력 차원으로 선형 변환하며, 학습 가능한 가중치(W)와 편향(b)을 통해 예측을 수행한다.
        # 선형 예측 (다른 예: softmax, sigmoid 등)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        """
        모델의 순전파를 정의한다.
        모델이 입력을 받아 예측값을 출력하는 방법을 정의한다.
        """
        # 모든 시점의 출력, 마지막 시점의 은닉 상태만 추출
        out, _ = self.gru(x)
        # out[:, -1, :]과 _는 거의 같은 정보를 담고 있지만, 형태가 달라서 실용적으로 out[:, -1, :]를 더 많이 쓴다.
        # 이 슬라이싱은 (batch_size, seq_len, hidden_size)의 shape 을 가진 3차원 텐서에서
        # [배치 차원, 시퀀스 차원, 피처 차원] : 모든 배치를 가져와 -1 시퀀스에서 마지막 시점 : 모든 피처를 가져와라.
        out = self.fc(out[:, -1, :])
        return out
    
    def save_model(self, path):
        """모델을 저장합니다."""
        # state_dict 를 활용해 모델의 가중치와 편향을 저장한다. 텐서만 포함하기 때문에 변경에 유연히 대처할 수 있다.
        torch.save(self.state_dict(), path)
    
    # 클래스 자체를 인자로 받아 동작하는 메서드
    # 해당 메소드를 이용하면
    # cls 를 인자로 받아, 설정 로딩과 함꼐 인스턴스 생성이 가능하다.
    @classmethod
    def load_model(cls, config_path, model_path):
        """저장된 모델의 가중치를 토대로, 모델 인스턴스를 반환한다."""
        model = cls(config_path)
        model.load_state_dict(torch.load(model_path))
        return model