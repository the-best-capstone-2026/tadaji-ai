# Tadanji AI 🍽️  
Food Image Classification Service

## 📌 Overview
**Tadanji AI**는 음식 이미지를 입력받아 해당 음식이 무엇인지 분류하는  
**이미지 분류 AI 추론 서비스**입니다.

Food-11 이미지 데이터셋(Kaggle)을 기반으로 학습된 딥러닝 모델을 사용하여,  
프론트엔드(React)에서 촬영/업로드한 음식 사진을 분석하고  
예측 결과(label, confidence, top-k)를 백엔드(Spring Boot)를 통해 반환합니다.

본 레포지토리는 **AI 모델 추론(Inference) 전용 서비스**로,  
FastAPI 기반의 독립적인 AI 서버 형태로 구성되어 있습니다.

---

## 🏗️ Architecture
[ React Frontend ]
     |
     v
[ Spring Boot Backend ]
     |
     v
[ FastAPI AI Server (this repo) ]
     |
     v
[ PyTorch Model ]


- 프론트엔드는 AI 서버에 직접 접근하지 않습니다.
- 백엔드는 이미지 업로드, 인증, 로그 관리 후 AI 서버에 추론 요청을 전달합니다.
- AI 서버는 **추론 결과만 반환**하며 비즈니스 로직을 포함하지 않습니다.

---

## 🧠 Model & Dataset
- **Dataset**: Food-11 Image Dataset (Kaggle)
- **Task**: Multi-class Image Classification (11 classes)
- **Framework**: PyTorch
- **Serving**: FastAPI

### Food-11 Classes
- Bread  
- Dairy Product  
- Dessert  
- Egg  
- Fried Food  
- Meat  
- Noodles/Pasta  
- Rice  
- Seafood  
- Soup  
- Vegetable/Fruit  

---

## 📂 Project Structure
adanji-ai/
├── src/
│ ├── app.py # FastAPI entry point
│ ├── inference.py # Image preprocessing & inference logic
│ ├── model_loader.py # Model loading utilities
│ ├── labels.py # Class label mapping
│
├── models/
│ └── model.pt # Trained PyTorch model
│
├── requirements.txt
├── Dockerfile
└── README.md


---

## 🚀 API Specification

### POST `/predict`
음식 이미지 파일을 입력받아 예측 결과를 반환합니다.

#### Request
- **Content-Type**: `multipart/form-data`
- **Body**
  - `file`: image file (jpg, png)

#### Response (Example)
```json
{
  "label": "pizza",
  "confidence": 0.92,
  "topK": [
    { "label": "pizza", "prob": 0.92 },
    { "label": "hamburger", "prob": 0.04 },
    { "label": "fried_rice", "prob": 0.02 }
  ]
}


