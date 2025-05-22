import cv2
import numpy as np
import os
from scipy.interpolate import CubicSpline  # CubicSpline 임포트 추가

# 카메라 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ 카메라를 열 수 없습니다.")
    exit()

# 저장 경로
save_folder = '/Users/parksungsu/Documents/python_opencv'
os.makedirs(save_folder, exist_ok=True)

output_path = os.path.join(save_folder, 'ball_video.mp4')
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'avc1')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("⚠️ 비디오 파일을 열 수 없습니다.")
    exit()

# 주황색 HSV 범위
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([15, 255, 255])

# 칼만 필터 (5차원: 중력 고려)
kalman = cv2.KalmanFilter(5, 2) 
# 5 = 5차원 상태 
# x, y = 위치
# vx, vy = 속도
# ay = y축 가속도(중력 등)

# 2 = 측정 벡터의 차원
# x, y 위치만 측정

# 이전 상태에서 다음 상태로의 변화(예측)를 모델링, 상태 전이 행렬
kalman.transitionMatrix = np.array([ 
    [1, 0, 1, 0, 0.5],   # x' = x + vx + 0.5*ay, 새로운 x 위치
    [0, 1, 0, 1, 0.5],   # y' = y + vy + 0.5*ay, 새로운 y 위치
    [0, 0, 1, 0, 0],     # vx' = vx, x축 속도 유지
    [0, 0, 0, 1, 1],     # vy' = vy + ay, y축 속도 변화
    [0, 0, 0, 0, 1]      # ay' = ay, 가속도 유지
], dtype=np.float32)

# 위치를 측정
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0, 0],     # x 위치만 측정
    [0, 1, 0, 0, 0]      # y 위치만 측정
], dtype=np.float32)

# 모델 노이즈 공분산 행렬
kalman.processNoiseCov = np.eye(5, dtype=np.float32) * 1e-2 
# 크기 5x5, 모델이 얼마나 불완전한지에 대한 추정
# 값이 클수록 모델 예측을 덜 신뢰하고, 측정값을 더 신뢰
# 1e-2는 적당한 수준의 모델 불확실성을 의미

# 측정 노이즈 공분산 행렬
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
# 크기 5x5, 측정이 얼마나 불완전한지에 대한 추정
# 값이 클수록 센서 측정을 덜 신뢰하고, 모델 예측을 더 신뢰
# 1e-1는 측정값에 0.1 정도의 오차가 있을 것이라 보는 것을 의미

trajectory = [] # 공의 이전 위치(예측 위치 포함)들을 저장하여 이동 경로를 기록
time_ahead = 10 # 10 프레임 분량의 미래 위치를 예측하여 시각화

# 추적기
tracker = cv2.legacy.TrackerKCF_create() 
# 한 번 검출한 객체(공)의 위치를 이후 프레임에서도 검출 없이 추적
# 칼만 필터와 별도로 이미지 기반 추적을 수행

tracking = False 
# 초기값으로 처음 공을 찾으면 True로 바꾸어 추적 시작
# 매 프레임마다 이 값에 따라 추적기 동작을 다르게 처리

kalman_initialized = False
# 칼만 필터 재설정(과거 상태 정보가 유지되지 않는 것)을 방지

while True:
    ret, img = cap.read()
    if not ret:
        print("⚠️ 프레임을 읽어올 수 없습니다.")
        break

    # 이미지 반전
    img = cv2.flip(img, 1) 

    # 가우시안 블러 적용 (노이즈 감소)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    # 값이 커지면 공의 경계가 흐릿해지고 추적 정확도가 떨어짐
    # 값이 작으면 노이즈 제거에 약함
    # 5, 7 중 테스트해봐야 함

    # 이미지 선명화 (Sharpening)
    kernel = np.array([[0, -1, 0], 
                        [-1, 5, -1], 
                        [0, -1, 0]]) # 기본 sharpen 필터
    sharpened_img = cv2.filter2D(blurred_img, -1, kernel)

    hsv = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2HSV) # 색 추출에 유리
    mask = cv2.inRange(hsv, lower_orange, upper_orange) # 범위 내 주황색 검출

    # 흰색 영역(주황색)을 더 정확하게 추출
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) # 잡음 제거 및 경계 강화
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # 구멍 채우기와 물체 연결

    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: # 마스크에서 검출된 모든 윤관선들에 대해 반복
        if cv2.contourArea(contour) > 5: 
        # 작은 영역(노이즈) 무시, 객체처럼 보이는 윤곽선만 처리
        # 50 ~ 100 정도로 올려서 잡음 제거 테스트해봐야 함

            M = cv2.moments(contour)
            if M["m00"] != 0: # 면적
                # 무게 중심(중심 좌표)를 계산
                cX = int(M["m10"] / M["m00"]) # x좌표의 평균, m10 = x좌표
                cY = int(M["m01"] / M["m00"]) # y좌표의 평균, m01 = y좌표

                if not tracking:
                    bbox = cv2.boundingRect(contour) # 주어진 윤곽선을 완전히 포함하는 가장 작은 직사각형을 구함
                    tracker.init(sharpened_img, bbox) # 객체의 위치를 기반으로 바운딩 박스를 만들고, 추적기 초기화
                    tracking = True # 추적 시작

                    # 칼만 필터 초기화(추적을 안정화하기 위한 초기 상태 설정)
                    kalman.statePre = np.array([[np.float32(cX)], # 검출된 물체의 중심 x좌표
                                                [np.float32(cY)], # 검출된 물체의 중심 y좌표
                                                [0],              # x 속도(vx)
                                                [0],              # y 속도(vy)
                                                [0]], np.float32) # 가속도
                    kalman.statePost = kalman.statePre.copy() # 측정 = 예측, 초기값
                    kalman_initialized = True # 예측과 측정 시작

                measurement = np.array([[np.float32(cX)], [np.float32(cY)]]) # 공이 측정된 위치를 칼만 필터에 넣기 위해 변환
                kalman.correct(measurement) 
                # 예측된 상태(statePre)와 실제 측정(measurement)을 비교하여 현재 상태를 더 정확하게 보정
                # 예측이 측정값과 가까워지고, 다음 예측(predict) 단계가 더 정확해짐

                cv2.circle(sharpened_img, (cX, cY), 5, (0, 255, 0), -1) # 초록색 점(측정된 위치 시각화)

    # 매 프레임 보정: 추적 결과를 보정값으로 사용
    if tracking:

        success, bbox = tracker.update(sharpened_img) # 현재 프레임에서 객체 추적

        if success: # 추적 성공
            (x, y, w, h) = [int(v) for v in bbox] # 바운딩 박스를 정수로 변환
            cv2.rectangle(sharpened_img, (x, y), (x + w, y + h), (0, 255, 0), 2) # 초록색 직사각형

            if kalman_initialized:
                center = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]]) # 중심 좌표 계산
                kalman.correct(center) # 칼만 필터의 측정값(measurement)으로 사용하여 필터를 보정

        else:       # 추적 실패
            if kalman_initialized: # 칼만 필터만 사용하여 위치를 예측
                prediction = kalman.predict()
                pred_x, pred_y = int(prediction[0, 0]), int(prediction[1, 0]) # 예측 x,y 좌표

                cv2.putText(sharpened_img, f"Predicted: ({pred_x}, {pred_y})", (pred_x, pred_y), # 추적 위치 좌표 및 예상 좌표
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) # 빨강색 글씨

    # 위치 예측 및 궤적 기록
    if kalman_initialized:
        prediction = kalman.predict() # 현재 상태(state)로부터 다음 위치 및 속도 등을 예측
        pred_x, pred_y = int(prediction[0, 0]), int(prediction[1, 0]) # 예측 x, y 위치

        # 속도(기울기)를 반영한 예측
        prev_velocity_x = int(prediction[2, 0])  # 이전 속도
        prev_velocity_y = int(prediction[3, 0])  # 이전 속도

        # 속도 변화량(기울기)을 반영하여 예측
        future_velocity_x = prev_velocity_x + int(prediction[4, 0])  # 예측된 속도 변화량
        future_velocity_y = prev_velocity_y + int(prediction[4, 0])  # 예측된 속도 변화량

        # 예측된 위치 업데이트
        pred_x += future_velocity_x
        pred_y += future_velocity_y

        if not (np.isnan(pred_x) or np.isnan(pred_y)):
            trajectory.append((pred_x, pred_y))

    # 궤적 그리기(파랑)
    for i in range(1, len(trajectory)):
        cv2.line(sharpened_img, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # 미래 궤적 예측 (Cubic Spline)
    if len(trajectory) >= 5:
        traj_np = np.array(trajectory)
        xs = traj_np[:, 0]
        ys = traj_np[:, 1]

        unique_xs, unique_indices = np.unique(xs, return_index=True)
        unique_ys = ys[unique_indices]

        if len(unique_xs) >= 2:
            cs = CubicSpline(unique_xs, unique_ys)

            # 햔재 예측 위치 pred_x부터 fps * time_ahead(약 10프레임치 거리)까지 Cubic Spline을 이용해 예측 궤적 계산
            future_xs = np.arange(pred_x, pred_x + np.sign(future_velocity_x) * fps * time_ahead, 2)
            future_points = []
            for fx in future_xs:
                fy = int(cs(fx))
                if 0 <= fx < sharpened_img.shape[1] and 0 <= fy < sharpened_img.shape[0]:
                    future_points.append((fx, fy))

            for i in range(1, len(future_points)):
                cv2.line(sharpened_img, future_points[i - 1], future_points[i], (0, 255, 255), 2)

    out.write(sharpened_img)
    cv2.imshow("Improved Tracking & Prediction", sharpened_img)

    #cv2.imshow("Test", mask)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        trajectory.clear()

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 비디오 파일이 저장되었습니다:", output_path)