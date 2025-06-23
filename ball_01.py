# 코드 흐름
# ------------------------------------------------------------------------------------------------------------
# 1. 주항색 탁구공 검출                                                                                            
# 2. 실제 경로를 초록색 곡선으로 표현                                                                                  
# 3. UKF(비선형 칼만 필터) + 등가속도 운동 공식(현재 위치) + (속도 * 시간) + (0.5 * 가속도 * 현재 위치 * 시간^2)을 사용한 궤적 예측
#    UKF를 사용하는 이유: 
#    (1) 칼만 필터(KF)보다 비선형적인 움직임(곡선)을 예측하는데 유리
#    (2) 구현이 단순하고 정확도가 높음
#    (3) 궤적을 예측하는 비선형 함수에 확률 분포를 의미히는 여러 개의 점들을 넣어, 점들의 결과를 평균/분산하여 예측 
#    등가속도 운동을 사용하는 이유: 
#    (1) 중력은 일정한 가속도(9.8, 영향을 주는 무게는 탁구공이기 때문에 무시)이기 때문에 대입하여 응용 가능 
#    (2) 탁구공은 짧은 시간에 빠르게 움직이기 때문에 공기저항, 회전력의 영향을 무시할 수 있음
#    (3) UKF와 궁합이 좋음(UKF는 기본 물리 모델이 예측 가능해야 필터가 잘 작동, 등가속도 운동은 구조 예측이 가능)                                                
# 4. 3초 후 예측 경로를 빨간색 곡선으로 표현
# 5. 영상 저장
# ------------------------------------------------------------------------------------------------------------

import cv2                                                # OpenCV 관련 라이브러리
import numpy as np                                        # 계산 관련 라이브러리
import os                                                 # 영상 저장 관련 라이브러리
from scipy.interpolate import CubicSpline                 # 곡선 관련 라이브러리
from filterpy.kalman import UnscentedKalmanFilter as UKF  # 비선형 칼만 필터(UKF)의 객체를 가져오는 코드
from filterpy.kalman import MerweScaledSigmaPoints        # 시그마 포인트 집합을 생성하는 도구

# 현재 영상 재생 = 30FPS(Frame Per Second), 1초에 30장의 이미지를 보여준다는 뜻
dt = 1 / 30 # 1초에 보여지는 30장의 이미지(프레임)를 탁구공의 궤적을 예측하는데 사용  

# 상태 전이 함수: 위치, 속도, 가속도 포함 (6차원 상태)
# 현재 상태 x에 대해, 시간 간격 dt가 지나면 다음 상태에 대해 예측
# 상태 벡터 x = [px, py, vx, vy, ax, ay], p = 위치, v = 속도, a = 가속도
# 등가속도 운동 방정식 기반
def fx(x, dt):
    F = np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0], # px(t+dt) = px + (vx*dt) + (0.5*ax*dt^2)
        [0, 1, 0, dt, 0, 0.5*dt**2],
        [0, 0, 1, 0, dt, 0],         # vx(t+dt) = vx + (ax*dt)
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],          # ax(t+dt) = ax(가속도는 일정하다고 가정)
        [0, 0, 0, 0, 0, 1]
    ])
    return F @ x                     # 행렬곱(@)을 이용해 기존 x에서 새로운 상태를 계산

# 관측 함수 (위치만 관측)
# 상태 x에서 px, py의 위치 정보만 측정
def hx(x):
    return x[:2]

# UKF 초기화  
points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=0)
# UKF에서 사용하는 시그마 포인트 생성기
# n=6,          상태 벡터의 차원
# alpha=0.1,    시그마 포인트 분포 범위를 제어(작을수록 중심 집중)
# beta=2.0,     가우시안 분포를 가정한 최적값(일반적으로 2)
# kappa=0,      보통 3-n으로 설정, 여기선 0
ukf = UKF(dim_x=6, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)
# UKF 객체 생성
# dim_x=6,      상태 벡터의 차원
# dim_z=2,      측정값의 차원
# fx=fx,        상태 전이 함수
# hx=hx,        관측 함수
# dt=dt,        시간 간격
# points=points 시그마 포인트 생성기

ukf.x = np.array([0, 0, 0, 0, 0, 0])        # 상태 벡터 초기값 설정(위치, 속도, 가속도 0으로 가정)
ukf.P *= 500                                # 상태 공분산 행렬 P 설정, 500은 초기 추정 오차를 크게 잡음(아직 아무것도 모르는 상태임을 반영)
ukf.R = np.diag([5, 5])                     # 측정 잡음 공분산 행렬 R 설정(측정값 px, py에 각각 +-5의 오차가 있다고 가정, 위치 측정할 때 약간의 픽셀 오차 반영)
q = 0.1                                     # 프로세스 잡음 공분산 행렬 Q 설정, 프로세스(모델)에 내재된 불확실성을 나타냄
ukf.Q = np.diag([q, q, q, q, q*10, q*10])   # 가속도는 예측이 더 어렵다고 가정해 *10 설정

# === 카메라 및 비디오 저장 설정 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

save_folder = '/Users/parksungsu/Documents/python_opencv'
os.makedirs(save_folder, exist_ok=True)
output_path = os.path.join(save_folder, 'ball_video.mp4')
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

# === 탁구공 HSV 범위 ===
lower_orange = np.array([5, 150, 100])
upper_orange = np.array([15, 255, 255])

real_trajectory = [] # 탁구공 궤적 좌표 저장 리스트

while True: # 영상 촬영 시작
    ret, img = cap.read() # 프레임을 읽고 이미지 받기
    if not ret:
        break

    img = cv2.flip(img, 1) # 이미지 반전

    # 모션 블러(움직임이 빨라지면 영상에서 흐릿하게 관찰) 대비를 위한 샤프닝 필터 적용
    sharpened_img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    # 노이즈 감소(값이 커지면 추적 정확도가 떨어지고 값이 작아지면 노이즈 제거에 약함)
    blurred_img = cv2.GaussianBlur(sharpened_img, (5, 5), 0)

    # 색 추출에 유리
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # 주황색 마스크 생성
    mask = cv2.inRange(hsv_img, lower_orange, upper_orange)

    # 마스크 정제(노이즈 제거)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 탁구공 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 탁구공 가운데 좌표값 초기화
    cx, cy = None, None

    for cnt in contours:
        area = cv2.contourArea(cnt)                                         # 윤곽선의 면적(픽셀 개수)를 계산
        if area > 400:                                                      # 픽셀값보다 작은 영역 무시
            x, y, w, h = cv2.boundingRect(cnt)                              # 윤곽선을 포함하는 최소 사각형의 x, y, w(너비), h(높이) 반환
            cx, cy = x + w // 2, y + h // 2                                 # 가운데 좌표 = 탁구공의 중심 추정
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)      # 탁구공의 영역에 파란색 사각형
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)                   # 탁구공의 중심에 초록색 점
            cv2.putText(img, f"Center: ({cx}, {cy})", (x, y - 10),          # 중심 좌표 흰색 텍스트
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            break

    ukf.predict() # UKF 추적

    # 탁구공의 궤적 추가
    if cx is not None and cy is not None:   # 공이 측정되면
        new_trajectory = np.array([cx, cy]) # UKF 필터에 새로운 측정값을 NumPy 배열로 생성
        ukf.update(new_trajectory)          # UKF 필터에 새로운 측정값을 넣어 상태를 보정(예측값이 틀렸다면, 이 측정값으로 위치, 속도, 가속도를 다시 정렬)
        real_trajectory.append((cx, cy))    # 현재 측정된 위치를 리스트에 추가(탁구공의 실제 이동 궤적을 저장)
        if len(real_trajectory) > 100:      # 공 궤적 좌표가 100개 이상이면
            real_trajectory.pop(0)          # 리스트 맨 앞 좌표(가장 오래된 위치) 제거

    # 실제 궤적 설정 
    if len(real_trajectory) >= 4:           # 4개 이상의 좌표로 안정적으로 곡선 보간(두 점 사이의 값을 자연스럽게 메워주는 기술) 가능 
        
        # 좌표 목록을 배열 형식으로 points 변수에 저장
        # ex) 4개의 좌표 (100, 200), (110, 210), (120, 215), (130, 220)
        points = np.array(real_trajectory)  # 배열로 저장, points.shape = (4, 2)
        t = np.arange(len(points))          # t = 4(0, 1, 2, 3), 프레임 순서(시간 역할) 기반의 변수

        x = points[:, 0] # 공의 가로(x) 좌표 값
        y = points[:, 1] # 공의 세로(y) 좌표 값

        # 프레임을 기준으로 곡선을 예측하는 함수 생성
        cs_x = CubicSpline(t, x) # 프레임에 따라 x 위치가 어떻게 변하는지 나타냄
        cs_y = CubicSpline(t, y) # 프레임에 따라 y 위치가 어떻게 변하는지 나타냄

        # 4개의 좌표 사이를 1000개의 시간 간격으로 나눠 더 부드러운 곡선을 그리는데 도움을 줌
        t_smooth = np.linspace(t.min(), t.max(), 1000)  # 곡선 좌표를 예측하는 1000개의 좌표를 얻을 수 있음
        x_smooth = cs_x(t_smooth)                       # x 좌표 예측
        y_smooth = cs_y(t_smooth)                       # y 좌표 예측

        # 탁구공 실제 궤적을 나타내는 초록색 곡선 그리기
        for i in range(1, len(t_smooth)):
            pt1 = (int(x_smooth[i - 1]), int(y_smooth[i - 1]))
            pt2 = (int(x_smooth[i]), int(y_smooth[i]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    # 예측 궤적 설정(3초 후, 90 프레임)
    future_steps = int(3.0 / dt)                                            # dt(30bps)를 사용, 3초 후를 예측하기 위해 future_steps(90 프레임)을 선언
    x_future = ukf.x.copy()                                                 # 위치, 속도, 가속도 0으로 설정
    predicted_trajectory = []                                               # 탁구공 궤적 예상 좌표 저장 리스트   

    for _ in range(future_steps):                                           # 예측 시작                                      
        x_future = fx(x_future, dt)                                         # 매 프레임(dt)마다 x_future를 다음 상태로 예측
        predicted_trajectory.append((int(x_future[0]), int(x_future[1])))   # 예측된 x, y 좌표를 정수로 변환 및 저장

    # 탁구공 예측 궤적을 나타내는 빨간색 곡선 그리기
    for i in range(1, len(predicted_trajectory)):                           
        cv2.line(img, predicted_trajectory[i - 1], predicted_trajectory[i], (0, 0, 255), 2)

    # 출력 및 저장

    # mask를 BGR로 변환하여 컬러 이미지로 출력
    mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # frame과 mask를 좌우로 붙이기
    combined_img = np.hstack((img, mask_img))

    # 출력
    cv2.imshow('Frame | Mask', combined_img)

    # ESC 키 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # 영상 저장
    out.write(img)

cap.release()
out.release()
cv2.destroyAllWindows()
print("\n✅ 비디오 파일이 저장되었습니다:", output_path)