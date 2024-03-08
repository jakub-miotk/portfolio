import OCRModel
import pytesseract


class PyTesseract(OCRModel):
    def __init__(self):
        super().__init__()

    def run(self, image):
        """Input image should be a binary image containing only license plate
        characters with adequate margin and border. Characters should be black and background white."""
        result = pytesseract.image_to_data(image, output_type='data.frame', lang='eng',
                                    config='--psm 8 -c tessedit_char_whitelist=0123456789QWERTYUIOPASDFGHJKLZXCVBNM -c '
                                           'load_system_dawg=false -c load_freq_dawg=false')
        if result["conf"].iloc[-1] == 0:  # Based on result analysis best results had confidence equal 0
            return str(result["text"].iloc[-1])
        else:
            return ''
