import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
# from tensorflow.keras.models import load_model # Keras 제거

# --- 1. PyTorch 모델 클래스 정의 ---
class GestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            batch_first=True # (batch, seq, feature)
        )
        
        self.relu1 = nn.ReLU()
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1] # 마지막 레이어의 hidden state
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x) # raw logits 반환
        return x

# --- 2. 모델 로드 및 설정 ---
actions = ['hello', 'come', 'spin']
seq_length = 30

input_size = 100
hidden_size = 64
num_classes = len(actions)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PyTorch:
model = GestureModel(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load('models/gesture_model.pt'))
model.eval()

# --- 3. MediaPipe 및 웹캠 설정 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []

print("Webcam running... Press 'q' to quit.")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for i, res in enumerate(result.multi_hand_landmarks):
            
            # --- 4. 피처 생성 ---
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # ... (v1, v2, v, angle 계산 코드는 동일) ...
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            angle = np.degrees(angle) # [15,]

            # 100번째 피처로 '손의 신뢰도 점수'를 추가합니다.
            # joint.flatten() (84) + angle (15) + confidence (1) = 100
            confidence = result.multi_handedness[i].classification[0].score
            d = np.concatenate([joint.flatten(), angle, [confidence]])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            # --- 5. 예측 부분 ---
            
            input_data_np = np.array(seq[-seq_length:], dtype=np.float32)
            input_data_np = np.expand_dims(input_data_np, axis=0) # (1, 30, 100)
            
            # 1. NumPy -> PyTorch Tensor로 변환 및 device로 전송
            input_tensor = torch.tensor(input_data_np, dtype=torch.float32).to(device)

            y_pred = None
            with torch.no_grad():
                outputs = model(input_tensor)
                outputs = torch.softmax(outputs, dim=1)
                y_pred = outputs.cpu().numpy().squeeze()
            
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            this_action='?' # 프레임의 기본 상태

            if conf < 0.9:
                this_action='NOTHING' # confidence 0.9 미만일 시 'NOTHING'으로 설정
                action_seq.clear()
            else:
                action=actions[i_pred]
                action_seq.append(action)

                if len(action_seq) >= 3:
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action

            cv2.putText(img, f'{this_action.upper()}', 
                        org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            print(this_action)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()