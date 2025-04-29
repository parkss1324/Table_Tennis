import cv2
import numpy as np
import os
from scipy.interpolate import CubicSpline

# 카메라 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ 카메라를 열 수 없습니다.")
    exit()

# 저장 경로 설정
save_folder = '/Users/parksungsu/Documents/python_opencv'
os.makedirs(save_folder, exist_ok=True)
output_path = os.path.join(save_folder, 'ball_video_with_table.mp4')
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'avc1')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 색상 범위 설정
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([15, 255, 255])

lower_blue = np.array([85, 30, 30])
upper_blue = np.array([145, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

# 칼만 필터 초기화
kalman = cv2.KalmanFilter(5, 2)
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0, 0.5],
    [0, 1, 0, 1, 0.5],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1]
], dtype=np.float32)
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
], dtype=np.float32)
kalman.processNoiseCov = np.eye(5, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

trajectory = []
time_ahead = 10
tracker = cv2.legacy.TrackerKCF_create()
tracking = False
kalman_initialized = False

# HSV 클릭 시 출력 함수
def show_hsv_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_val = param[y, x]
        print(f"👉 HSV at ({x}, {y}) = {hsv_val}")

cv2.namedWindow("Ball & Table Tracking")

while True:
    ret, img = cap.read()
    if not ret:
        print("⚠️ 프레임을 읽어올 수 없습니다.")
        break

    # 조명 보정
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.flip(img, 1)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(blurred_img, -1, kernel)

    hsv = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2HSV)

    # HSV 클릭 콜백
    cv2.setMouseCallback("Ball & Table Tracking", show_hsv_on_click, hsv)

    # 탁구대(파란색, 흰색 경계 포함) 검출
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(blue_mask, white_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                cv2.drawContours(sharpened_img, [approx], -1, (255, 100, 0), 3)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(sharpened_img, "Ping Pong Table", (cX - 50, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    # 주황색 공 검출
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours_orange, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_orange:
        if cv2.contourArea(contour) > 1:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if not tracking:
                    bbox = cv2.boundingRect(contour)
                    tracker.init(sharpened_img, bbox)
                    tracking = True
                    kalman.statePre = np.array([[np.float32(cX)], [np.float32(cY)], [0], [0], [0]], np.float32)
                    kalman.statePost = kalman.statePre.copy()
                    kalman_initialized = True

                measurement = np.array([[np.float32(cX)], [np.float32(cY)]])
                kalman.correct(measurement)
                cv2.circle(sharpened_img, (cX, cY), 5, (0, 255, 0), -1)

    if tracking:
        success, bbox = tracker.update(sharpened_img)
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(sharpened_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if kalman_initialized:
                center = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
                kalman.correct(center)
        else:
            if kalman_initialized:
                prediction = kalman.predict()
                pred_x, pred_y = int(prediction[0, 0]), int(prediction[1, 0])
                cv2.putText(sharpened_img, f"Predicted: ({pred_x}, {pred_y})", (pred_x, pred_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 궤적 저장
    if kalman_initialized:
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0, 0]), int(prediction[1, 0])
        prev_velocity_x = int(prediction[2, 0])
        prev_velocity_y = int(prediction[3, 0])
        future_velocity_x = prev_velocity_x + int(prediction[4, 0])
        future_velocity_y = prev_velocity_y + int(prediction[4, 0])
        pred_x += future_velocity_x
        pred_y += future_velocity_y
        if not (np.isnan(pred_x) or np.isnan(pred_y)):
            trajectory.append((pred_x, pred_y))

    # 현재 궤적 시각화
    for i in range(1, len(trajectory)):
        cv2.line(sharpened_img, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # 미래 궤적 예측 (스플라인 보간)
    if len(trajectory) >= 5:
        traj_np = np.array(trajectory)
        xs = traj_np[:, 0]
        ys = traj_np[:, 1]
        unique_xs, unique_indices = np.unique(xs, return_index=True)
        unique_ys = ys[unique_indices]
        if len(unique_xs) >= 2:
            cs = CubicSpline(unique_xs, unique_ys)
            future_xs = np.arange(pred_x, pred_x + np.sign(future_velocity_x) * fps * time_ahead, 2)
            future_points = []
            for fx in future_xs:
                fy = int(cs(fx))
                if 0 <= fx < sharpened_img.shape[1] and 0 <= fy < sharpened_img.shape[0]:
                    future_points.append((fx, fy))
            for i in range(1, len(future_points)):
                cv2.line(sharpened_img, future_points[i - 1], future_points[i], (0, 255, 255), 2)

    out.write(sharpened_img)
    cv2.imshow("Ball & Table Tracking", sharpened_img)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        trajectory.clear()

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 비디오 파일이 저장되었습니다:", output_path)
