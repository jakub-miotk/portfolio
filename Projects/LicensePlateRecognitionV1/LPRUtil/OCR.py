import re
from LPImage import is_image_empty, prepare_lp_for_ocr


def execute_ocr(cut_out_lp, ocr_model, ih=44, mh=3, mw=8, btb=4, blr=6):
    """Parameters default values were found during experiments, they yield best OCR results. Parameter's full names:
    image height, margin width, margin height, border top/bottom, border left/right."""
    if not is_image_empty(cut_out_lp):
        return ''
    cropped_lp = prepare_lp_for_ocr(cut_out_lp, ih, mw, mh, btb, blr)  # Preprocessing
    if cropped_lp is None:
        return ''
    lp_number = ocr_model.run(cropped_lp)
    lp_number = process_ocr_result(lp_number)
    return lp_number


def digit_to_letter(match_obj):
    if match_obj.group(0) == '0':
        return 'O'
    elif match_obj.group(0) == '1':
        return 'I'
    elif match_obj.group(0) == '2':
        return 'Z'
    elif match_obj.group(0) == '5':
        return 'S'
    elif match_obj.group(0) == '7':
        return 'Z'
    elif match_obj.group(0) == '8':
        return 'B'
    else:
        return match_obj.group(0)


def letter_to_digit(match_obj):
    if match_obj.group(0) == 'B':
        return '8'
    if match_obj.group(0) == 'D':
        return '0'
    elif match_obj.group(0) == 'I':
        return '1'
    elif match_obj.group(0) == 'O':
        return '0'
    elif match_obj.group(0) == 'Z':
        return '2'
    else:
        return match_obj.group(0)


def process_ocr_result(ocr_result):
    # Remove characters that cannot appear on a license plate
    ocr_result = re.sub(r'[^A-Z0-9 ]', '', ocr_result)
    # Remove characters I and 1 on first position, as those are almost always a result of preprocessing error,
    # which is difficult to fix
    ocr_result = re.sub(r'^I1', '', ocr_result)
    # Change digits to letters in first 2 characters, as there can only be characters in those positions
    ocr_result = re.sub(r'(?:(?<=^)|(?<=^.))\d', digit_to_letter, ocr_result)
    # Change certain letters to digits in positions 3 to 8. This change is based on license plate regulations in Poland
    ocr_result = re.sub(r'(?:(?<=...)|(?<=....)|(?<=.....)|(?<=......)|(?<=.......))[BDIOZ]',
                        letter_to_digit, ocr_result)
    # Remove unnecessary spaces
    ocr_result = re.sub(r' ', '', ocr_result)
    return ocr_result
