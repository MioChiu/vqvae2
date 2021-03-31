import cv2

img1 = cv2.imread('0_ori.png')
img2 = cv2.imread('0_out.png')
img3 = img1 - img2
cv2.imwrite('0_diff.png', img3)
