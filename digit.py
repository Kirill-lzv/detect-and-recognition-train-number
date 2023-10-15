import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised_image = cv2.fastNlMeansDenoising(thresholded_image, h=10)
    alpha = 1.5
    enhanced_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=0)
    cv2.imshow('1', enhanced_image)
    cv2.waitKey(100)
    return enhanced_image


image_path = 'img_0.jpg'
preprocessed_image = preprocess_image(image_path)

recognized_text = pytesseract.image_to_string(preprocessed_image,
                                              config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

print(f'Распознанные цифры: {recognized_text}')
