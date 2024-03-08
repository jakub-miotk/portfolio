import numpy as np
import cv2
from queue import Queue as qQueue


# This is only a selected fragment of LPRecognition class
class LPRecognition:
    def __init__(self):
        pass

    def distribute_new_track_windows_to_processes(self, track_windows):
        """Distribute new track windows to tracker processes based on the load they are under."""
        loads = np.array([trackers_process.load for trackers_process in self.trackers_processes], dtype=np.uint8)
        distributed_track_windows = [[] for _ in range(len(self.trackers_processes))]
        for track_window in track_windows:
            if np.min(loads) < self.max_trackers_per_process:  # Send new tracker to sub_process with the lowest load
                min_load_idx = np.argmin(loads)
                distributed_track_windows[min_load_idx].extend([track_window])
                loads[min_load_idx] += 1
        return distributed_track_windows

    def detect_lp_in_video(self, path):
        cap = cv2.VideoCapture(path)
        self.init_sub_processes()
        frame_q = qQueue()
        frames_to_put_in_q = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if frame_q.qsize() + len(frames_to_put_in_q) < self.frame_q_size:  # First fill buffer queue
                frame_q.put(frame)
                continue
            # If all processes are ready start a new update cycle
            if not self.all_updating():  # max 0.00018s
                self.send_frame_to_detector(frame)
                self.update_trackers(frame)
            if self.any_updating():  # max 0.00001s
                frames_to_put_in_q.extend([frame])
            # Check if any process has sent any new information
            self.receive_detector_data()  # max 0.00008s
            self.receive_tracker_data()  # max 0.00008s
            # If all trackers finished updating positions send new trackers to be initialized
            if self.received_all_data():  # max 0.007s
                track_windows = self.get_track_windows()
                self.start_new_trackers(self.get_lp_bounding_boxes(), track_windows)
                self.save_results(self.get_ocr_results())
                if self.show_detections:
                    self.draw_tracked_objects(frames_to_put_in_q[0], track_windows,
                                              self.detection_color, self.detection_thickness)
                self.load_frames_to_q(frames_to_put_in_q, frame_q)
                frames_to_put_in_q = []
            popped_frame = frame_q.get(timeout=5)
            if self.show_detections:
                cv2.namedWindow('License plate detections', cv2.WINDOW_NORMAL)
                cv2.imshow('License plate detections', popped_frame)
        self.kill_detector_process()
        self.close_trackers()
