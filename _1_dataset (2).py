import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image 
import json
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms




def image_preprocessing(img, how='original'):
    """
    Parameters:
        img: RGB 형식의 입력 이미지
        how: 전처리 방법
    Returns:
        RGB 형식의 전처리된 이미지
    """
    # RGB -> BGR 변환 (OpenCV 함수들을 위해)
    if how != 'original':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if how == 'original':
        transformed_img = img
       # channels = cv2.split(img)
       # denoised_channels = [cv2.fastNlMeansDenoising(channel, None, 10, 7, 21) for channel in channels]
       # transformed_img = cv2.merge(denoised_channels)
        
    elif how == 'denoise_per_channel':
        channels = cv2.split(img)
        denoised_channels = [cv2.fastNlMeansDenoising(channel, None, 10, 7, 21) for channel in channels]
        transformed_img = cv2.merge(denoised_channels)
    
    elif how == 'histeq_per_channel':
        channels = cv2.split(img)
      #  denoised_channels = [cv2.fastNlMeansDenoising(channel, None, 10, 7, 21) for channel in channels]
        histeq_channels = [cv2.equalizeHist(channel) for channel in channels]
        transformed_img = cv2.merge(histeq_channels)
    
    elif how == 'clahe_per_channel':
        channels = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_channels = [clahe.apply(channel) for channel in channels]
        transformed_img = cv2.merge(clahe_channels)
    
    elif how == 'gamma_cor_channel':
        channels = cv2.split(img)
        gamma_channels = [np.array(255 * (channel / 255) ** 0.5, dtype='uint8') for channel in channels]
        transformed_img = cv2.merge(gamma_channels)
    
    elif how == 'threshold_otsu':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, t_otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        transformed_img = cv2.cvtColor(t_otsu, cv2.COLOR_GRAY2BGR)
    
    elif how == 'threshold_adapted_mean':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        transformed_img = cv2.cvtColor(mean_img, cv2.COLOR_GRAY2BGR)
    
    elif how == 'threshold_adapted_gaus':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        transformed_img = cv2.cvtColor(gaus_img, cv2.COLOR_GRAY2BGR)
    
    else:
        raise ValueError(f"Unsupported preprocessing method: {how}")
    
    # BGR -> RGB 변환 (결과를 다시 RGB로)
    if how != 'original':
        transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
    
    return transformed_img


def add_noise(image):
    noise = torch.randn_like(image) * 0.1
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


class CustomDataset(Dataset):
    def __init__(self, image_path_list, label_list, input_size=None, how='original', train=False, model_ckpt=None):
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.how = how
        self.train = train
        self.input_size = input_size

        if train:
            self.mean, self.std = self._calculate_dataset_statistics(model_ckpt)
        else:
            self.mean, self.std = self._load_dataset_statistics(model_ckpt)

    def _calculate_dataset_statistics(self, model_ckpt=None):
        """데이터셋의 평균과 표준편차를 계산"""
        means = []
        stds = []
        
        # 진행 상황을 보여주기 위한 tqdm 사용
        for path in tqdm(self.image_path_list):
            img = cv2.imread(path)            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = image_preprocessing(img, self.how)
            
            img = cv2.resize(img, (self.input_size, self.input_size))
            img = img.astype(np.float32) / 255.0  # [0,1] 범위로 정규화
            
            means.append(img.mean(axis=(0,1)))
            stds.append(img.std(axis=(0,1)))
            
        # 전체 데이터셋의 평균과 표준편차 계산
        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)
        
        # 모델 체크포인트 경로가 제공된 경우 JSON으로 저장
        if model_ckpt is not None:
            stats = {
                'mean': mean.tolist(),
                'std': std.tolist(),
                'preprocessing_method': self.how,
                'input_size': self.input_size
            }
            json_path = os.path.join(model_ckpt, 'dataset_statistics.json')
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=4)
        
        return mean, std
    
    def _load_dataset_statistics(self, model_ckpt):
        """저장된 train set의 통계값을 로드"""
        json_path = os.path.join(model_ckpt, 'dataset_statistics.json')
        with open(json_path, 'r') as f:
            stats = json.load(f)
        return np.array(stats['mean']), np.array(stats['std'])
    
    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB 변환
        img = image_preprocessing(img, self.how) # 전처리

        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0 # Normalize to [0,1] range
        img = torch.from_numpy(img).permute(2, 0, 1) # Convert to tensor and permute dimensions
        normalize = transforms.Normalize(mean=self.mean, std=self.std) # Dataset specific normalization
        img = normalize(img)
        return img

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = self.get_image(image_path)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image_path, image, label
        return image_path, image
        
    def __len__(self):
        return len(self.image_path_list)

    
def show_image_from_dataset(dataset, index, model_ckpt):
    image_path, image, label = dataset[index]
    image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    
    # Denormalize using dataset-specific statistics
    mean = torch.tensor(dataset.mean)
    std = torch.tensor(dataset.std)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    
    plt.figure(figsize=(4, 4))
    
    plt.imshow(image)
    plt.title(f'Processed (Label: {label})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_ckpt}/ex_img_trainds.png', bbox_inches='tight', pad_inches=0)
    plt.show()
  
