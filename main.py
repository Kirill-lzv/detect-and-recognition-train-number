import cv2
import tensorflow as tf
import numpy as np
# from google.colab.patches import cv2_imshow
import pytesseract

desired_height, desired_width = 600, 600

model = tf.keras.models.load_model("mod_212_600.h5")

image = cv2.imread("42324988.jpg")

img = image
img_1 = image
image_r = cv2.resize(image, (desired_width, desired_height))

# cv2.imshow(image_r)
image = image_r / 255.0
coordinates = model.predict(np.array([image]))
print(coordinates[0])

x, y, width, height = coordinates[0]
x, y, width, height = int(x), int(y), int(width), int(height)
print(x, y, width, height, "coordinates")
# image_with_rectangle = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
# cv2_imshow(image_with_rectangle)
crop_img = img_1[y:y+height, x:x+width]
crop_img1 = img_1[y-20:y+height+20, x-70:x+width]

cv2.imshow('', crop_img)
cv2.imshow('', crop_img1)

cv2.waitKey(33)


cv2.destroyAllWindows()

def preprocess_image(image1):
    # image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised_image = cv2.fastNlMeansDenoising(thresholded_image, h=10)
    alpha = 1.5
    enhanced_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=0)
    cv2.imshow('1', enhanced_image)
    cv2.waitKey(100)
    return enhanced_image


image1 = crop_img1
preprocessed_image = preprocess_image(image1)

recognized_text = pytesseract.image_to_string(preprocessed_image,
                                              config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

print(f'Распознанные цифры: {recognized_text}')

