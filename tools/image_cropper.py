import cv2
import os
from crewai.tools import tool

class ImageCropper:
    def crop_and_save(self, input_path: str, output_path: str, x: int, y: int, width: int, height: int) -> bool:
        if not os.path.exists(input_path):
            return False
        image = cv2.imread(input_path)
        if image is None:
            return False
        cropped_image = image[y:y+height, x:x+width]
        return cv2.imwrite(output_path, cropped_image)

# İşte main.py'nin aradığı ve Llama 3'ün çok sevdiği o İngilizce araç ismi:
@tool("crop_image_tool") 
def crop_image_tool(input_path: str, output_path: str, x: int, y: int, width: int, height: int) -> str:
    """
    Crops an image from given x, y coordinates with width and height.
    Saves the result to output_path.
    """
    cropper = ImageCropper()
    success = cropper.crop_and_save(input_path, output_path, x, y, width, height)
    if success:
        return f"İşlem Başarılı: Görüntü {output_path} adresine kaydedildi."
    return "Hata: Görüntü kırpılamadı."