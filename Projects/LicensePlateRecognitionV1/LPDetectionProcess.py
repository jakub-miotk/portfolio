from multiprocessing import Process
import cv2
import LPRUtil.LPImage as LPRui
import LPRUtil.Contour as LPRuc
import LPRUtil.Rect as LPRur


# Full class of a subprocess whose task is to find new license plates in a frame.
class LPDetectionProcess(Process):

    def __init__(self, q_to, q_from, proportions, proportions_sigma, max_par_angle, max_perp_angle):
        super(LPDetectionProcess, self).__init__()
        self.daemon = True
        self.lp_bounding_boxes = []
        self.frame = None
        self.q_to = q_to  # Queue to send data to the main process
        self.q_from = q_from  # Queue to receive data from the main process
        self.proportions = proportions
        self.proportions_sigma = proportions_sigma
        self.max_par_angle = max_par_angle  # parallel
        self.max_perp_angle = max_perp_angle  # perpendicular

    def run(self):  # Main process loop
        self._send_status(updating=False)
        while True:
            kill_process = self._receive_frame()
            if kill_process:
                break
            binary_frame = LPRui.binarize(self.frame)
            self.lp_bounding_boxes = self.find_lp_in_binary_image(binary_frame)
            self.lp_bounding_boxes = LPRur.merge_overlapping_rects(self.lp_bounding_boxes)
            self._send_found_lps()

    def _send_status(self, updating):
        response = {'updating': updating}
        self.q_to.put(response)

    def _receive_frame(self):
        task_data = self.q_from.get(timeout=10)
        self.frame = task_data['frame']
        if task_data['kill_process']:
            return True
        return False

    def _send_found_lps(self):
        response = {'lp_bounding_boxes': self.lp_bounding_boxes, 'updating': False}
        self.q_to.put(response)
        self.lp_bounding_boxes = []

    def find_lp_in_binary_image(self, binary_frame):
        contours = LPRuc.get_contours(binary_frame)  # Get all contours from a frame
        lp_bounding_boxes = []
        for c in contours:
            contour_perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * contour_perimeter, True)
            # If approximation of a contour has a rectangular shape and fits into proportion requirements
            # it's most likely a license plate
            if self.check_lp_shape(approx):
                lp_bounding_boxes.extend([cv2.boundingRect(c)])
        return lp_bounding_boxes

    def check_lp_shape(self, contour):
        """Assess contour's similarity to a rectangle and it's proportions."""
        if len(contour) != 4:
            return False
        contour = LPRuc.preprocess_contour_shape(contour)  # Reshape contour data
        # Reorder contour points to [[top-left], [bottom-left], [top-right], [bottom-right]]
        contour = LPRuc.reorder_contour_points(contour)
        if LPRuc.check_contour_proportions(contour, self.proportions, self.proportions_sigma):
            return LPRuc.check_contour_angles(contour, self.max_par_angle, self.max_perp_angle)
        return False
