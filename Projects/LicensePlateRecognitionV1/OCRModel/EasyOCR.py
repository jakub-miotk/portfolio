import OCRModel
import easyocr


class EasyOCR(OCRModel):
    def __init__(self):
        super().__init__()
        self.model = easyocr.Reader(['en'])

    def run(self, image):
        """Input image should be a binary image containing only license plate
        characters with adequate margin and border. Characters should be black and background white."""
        ocr_result = self.model.readtext(image, allowlist='0123456789QWERTYUIOPASDFGHJKLZXCVBNM')
        if len(ocr_result) > 0:
            text = ocr_result[0][1]
        else:
            text = ''
        return text
