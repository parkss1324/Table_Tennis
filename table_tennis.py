import cv2
import numpy as np
import os

# 카메라 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ 카메라를 열 수 없습니다.")
    exit()

# 저장 경로
save_folder = '/Users/parksungsu/Documents/python_opencv'
os.makedirs(save_folder, exist_ok=True)

output_path = os.path.join(save_folder, 'table_tennis.mp4')
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'avc1')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("⚠️ 비디오 파일을 열 수 없습니다.")
    exit()

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.resize(image, (frame_width, frame_height))
    mask = np.zeros(image.shape, np.uint8)

    # ▶️ 흰색 마스크 추출
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # ▶️ 흰색 마스크 기반으로만 Canny 수행
    masked_image = cv2.bitwise_and(image, image, mask=white_mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # ▶️ 흰색 영역에서만 엣지 사용
    canny = cv2.bitwise_and(canny, white_mask)

    # 수직선 검출
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    remove_horizontal = cv2.morphologyEx(canny, cv2.MORPH_OPEN, vertical_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate_vertical = cv2.morphologyEx(remove_horizontal, cv2.MORPH_CLOSE, kernel, iterations=5)

    lines = cv2.HoughLinesP(dilate_vertical, 1, np.pi / 180, 100, 10, 150)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # 수평선 검출
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    remove_vertical = cv2.morphologyEx(canny, cv2.MORPH_OPEN, horizontal_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate_horizontal = cv2.morphologyEx(remove_vertical, cv2.MORPH_CLOSE, kernel, iterations=3)

    lines = cv2.HoughLinesP(dilate_horizontal, 1, np.pi / 180, 100, 10, 300)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # 마스크에서 외곽선 검출
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 2)

    # 결과 디스플레이
    cv2.imshow('Table Detection', image)
    #cv2.imshow('White Mask', white_mask)
    #cv2.imshow('Canny in White Area', canny)

    # 프레임 저장
    out.write(image)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 비디오 파일이 저장되었습니다:", output_path)