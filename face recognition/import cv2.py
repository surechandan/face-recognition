import cv2

img = cv2.imread(r"C:\Users\surec\Downloads\dataset\dataset\faces\Deepika\face_2.jpg", 0)

print(img)   # CHECK IMAGE LOADING

resized = cv2.resize(img, (100, 100))

cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
