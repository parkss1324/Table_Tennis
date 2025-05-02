import cv2
import numpy as np

# 이미지 경로 설정
image_path = '/Users/parksungsu/Documents/python_opencv/table_tennis.png'

# 이미지 불러오기
img = cv2.imread(image_path)

# 이미지가 정상적으로 불러와졌는지 확인
if img is None:
    print("⚠️ 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

# 흰색 영역 마스크 (BGR 기준으로 흰색 근처 설정)
lower_white = np.array([150, 150, 150], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
white_mask = cv2.inRange(img, lower_white, upper_white)

# 흰색 부분을 초록색으로 변경
img[white_mask > 0] = [0, 255, 0]

# 그레이스케일 변환 후 엣지 검출
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 허프 변환으로 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

# 검출된 직선들을 초록색 선으로 그림
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 결과 이미지 보여주기
cv2.imshow('White Areas and Lines as Green', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
