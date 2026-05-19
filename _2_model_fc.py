import torch
import torch.nn as nn

def create_fc_model(input_size=522, n_layers=1, first_hidden=128, hidden_sizes=None, use_dropout=False, dropout_rate=0.3):
    """
    파라미터화된 FC 레이어 모델 생성 함수
    
    Args:
        input_size: 입력 특성 벡터 크기
        n_layers: 레이어 수
        first_hidden: 첫 번째 레이어의 뉴런 수
        hidden_sizes: 중간 레이어들의 뉴런 수 리스트 (None이면 빈 리스트로 초기화)
        use_dropout: 드롭아웃 사용 여부
        dropout_rate: 드롭아웃 비율
    
    Returns:
        nn.Module: 구성된 FC 모델
    """
    if hidden_sizes is None:
        hidden_sizes = []
    
    layers = []
    
    if n_layers == 1:
        # 단순 선형 레이어
        return nn.Linear(input_size, 2)
    else:
        # 레이어 구성
        layers.append(nn.Linear(input_size, first_hidden))
        layers.append(nn.ReLU())
        
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        
        # 중간 레이어들
        current_size = first_hidden
        for size in hidden_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            current_size = size
        
        # 출력 레이어
        layers.append(nn.Linear(current_size, 2))
        
        return nn.Sequential(*layers)