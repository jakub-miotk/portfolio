import numpy as np
import pandas as pd


class LPTracker:
    def __init__(self):
        pass

    def merge_ocr_results(self):
        lp_length = self._get_lp_length()  # Get most probable length of lp
        # Create a DF to gather pseudo probabilities of characters appearing at certain positions in lp
        number_range = np.arange(48, 58)  # 0 - 9
        letter_range = np.arange(65, 91)  # A - Z
        char_range = np.concatenate((number_range, letter_range))
        chars = [chr(char_ascii) for char_ascii in char_range]
        character_prob = pd.DataFrame(index=chars,
                                      columns=np.arange(lp_length),
                                      data=np.zeros((char_range.size, lp_length)))
        # Calculate pseudo probabilities for characters
        for ocr_output in self.ocr_outputs:
            ocr_len = len(ocr_output)
            for i, character in enumerate(ocr_output):
                if 0 <= i - 2 < lp_length:
                    character_prob.loc[character, i - 2] += 0.05 / (np.abs(lp_length - ocr_len) + 1)
                if 0 <= i - 1 < lp_length:
                    character_prob.loc[character, i - 1] += 0.1 / (np.abs(lp_length - ocr_len) + 1)
                if i < lp_length:
                    character_prob.loc[character, i] += 0.7 / (np.abs(lp_length - ocr_len) + 1)
                if i + 1 < lp_length:
                    character_prob.loc[character, i + 1] += 0.1 / (np.abs(lp_length - ocr_len) + 1)
                if i + 2 < lp_length:
                    character_prob.loc[character, i + 2] += 0.05 / (np.abs(lp_length - ocr_len) + 1)
        # Get most 'probable' lp number
        lp_number = ''.join([c for c in character_prob.idxmax()])
        return lp_number
    
    def _get_lp_length(self):
        lp_lengths = np.array([len(ocr_output) for ocr_output in self.ocr_outputs])
        length_counts = np.zeros(self.max_lp_length - self.min_lp_length + 1)
        for length in range(self.min_lp_length, self.max_lp_length + 1):
            length_counts[length - self.min_lp_length] = np.where(lp_lengths == length)[0].size
        return np.argmax(length_counts) + self.min_lp_length