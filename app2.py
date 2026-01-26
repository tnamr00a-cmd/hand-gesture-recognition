#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import itertools
from collections import Counter, deque
from threading import Thread
import time
import math

# Thay thế keyboard bằng cách dùng cv2.waitKey để đồng bộ tốt hơn với cửa sổ hiển thị
# import keyboard (Đã loại bỏ để giảm độ trễ input không cần thiết)

import cv2 as cv
import mediapipe as mp
import numpy as np

# Các thư viện âm thanh
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, EDataFlow, ERole

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    args = parser.parse_args()
    return args


# --- CLASS ĐA LUỒNG CAMERA (Tăng tốc FPS cực mạnh) --- 
class WebcamStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv.VideoCapture(src)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        # Tắt bộ đệm để giảm độ trễ hình ảnh
        self.stream.set(cv.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Trả về frame mới nhất hiện có
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.stream.release()


def main():
    args = get_args()

    # --- KHỞI TẠO ÂM THANH ---
    volume = None
    min_vol, max_vol = -65.25, 0.0
    try:
        devices = AudioUtilities.GetDeviceEnumerator()
        interface = devices.GetDefaultAudioEndpoint(EDataFlow.eRender.value, ERole.eMultimedia.value)
        volume = cast(interface.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))
        vol_range = volume.GetVolumeRange()
        min_vol, max_vol = vol_range[0], vol_range[1]
    except Exception as e:
        print(f"Lỗi âm thanh: {e}")

    # --- KHỞI TẠO MEDIAPIPE ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=0 # Giảm độ phức tạp model để tăng FPS (0=Lite, 1=Full)
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Đọc Labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Lịch sử tọa độ
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Biến trạng thái
    mode = 0
    number = -1
    control_enabled = True
    active_hand_id = None
    
    # Biến làm mượt âm thanh
    smooth_vol = 0
    last_set_vol = -100 # Để kiểm tra tránh spam lệnh set volume
    vol_alpha = 0.5 # Hệ số làm mượt

    # --- KHỞI ĐỘNG CAMERA ĐA LUỒNG ---
    cap = WebcamStream(src=args.device, width=args.width, height=args.height).start()
    
    print("Hệ thống đã sẵn sàng. Nhấn 'M' để bật/tắt điều khiển, 'ESC' để thoát.")

    while True:
        fps = cvFpsCalc.get()

        # Xử lý phím bấm (Dùng cv.waitKey thay vì keyboard lib để nhanh hơn)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32: # Space -> Đổi chế độ ghi log nếu cần (tuỳ chọn)
            pass 
        elif key == ord('m'):
            control_enabled = not control_enabled
        
        # Mode selection logic (giữ nguyên)
        number, mode = select_mode(key, mode)

        # Đọc frame từ luồng riêng
        ret, frame = cap.read()
        if not ret:
            continue  # Nếu chưa có frame, bỏ qua vòng lặp này

        # Xử lý ảnh: Flip -> Convert RGB
        # Lưu ý: Không dùng deepcopy nữa, vẽ trực tiếp lên frame sau khi lật
        frame = cv.flip(frame, 1) 
        debug_image = frame # Tham chiếu trực tiếp, không copy để tiết kiệm RAM/CPU
        
        # MediaPipe cần RGB, OpenCV dùng BGR
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Gắn cờ writeable = False để tăng tốc xử lý MP một chút
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True # Trả lại trạng thái

        if results.multi_hand_landmarks:
            current_hand_ids = [h.classification[0].index for h in results.multi_handedness]
            
            # Reset active hand nếu tay đó biến mất
            if active_hand_id not in current_hand_ids:
                active_hand_id = None

            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                hand_id = handedness.classification[0].index
                
                # Logic chọn tay điều khiển (Lock-on)
                if active_hand_id is None:
                    active_hand_id = hand_id

                # Tính toán bounding box & landmark
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # --- ĐIỀU KHIỂN ÂM THANH (Đã tối ưu) ---
                if control_enabled and hand_id == active_hand_id:
                    # Chỉ xử lý khi đủ điểm landmark
                    x1, y1 = landmark_list[4]
                    x2, y2 = landmark_list[8]
                    
                    # Tính khoảng cách
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Vẽ trực quan
                    cv.circle(debug_image, (x1, y1), 8, (255, 0, 255), cv.FILLED)
                    cv.circle(debug_image, (x2, y2), 8, (255, 0, 255), cv.FILLED)
                    cv.line(debug_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Mapping khoảng cách sang Volume
                    target_vol = np.interp(distance, [30, 150], [min_vol, max_vol])
                    
                    # Làm mượt giá trị volume (Exponential Moving Average)
                    # Nếu smooth_vol chưa có giá trị, gán bằng target luôn
                    smooth_vol = target_vol if smooth_vol == 0 else (vol_alpha * target_vol) + ((1 - vol_alpha) * smooth_vol)
                    
                    # Chỉ set volume nếu thay đổi đủ lớn (> 0.5dB) để tránh spam hệ thống
                    if abs(smooth_vol - last_set_vol) > 0.5:
                        volume.SetMasterVolumeLevel(smooth_vol, None)
                        last_set_vol = smooth_vol

                    vol_per = np.interp(smooth_vol, [min_vol, max_vol], [0, 100])
                    cv.putText(debug_image, f"VOL: {int(vol_per)}%", (brect[0], brect[1]-30),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # ---------------------------------------

                # Pre-process landmark & Point history
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image.shape[1], debug_image.shape[0], point_history)
                
                # Logging CSV
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                # Gesture Classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger Gesture Classification
                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Vẽ UI
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            # Không phát hiện tay
            point_history.append([0, 0])
            # Reset lock tay nếu mất dấu quá lâu (tuỳ chọn, ở đây giữ active_id để đỡ bị nhảy)

        # Vẽ Menu & FPS
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_menu_status(debug_image, control_enabled)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    # Cleanup
    cap.release()
    cv.destroyAllWindows()


# --- CÁC HÀM HỖ TRỢ (GIỮ NGUYÊN HOẶC TỐI ƯU NHẸ) ---

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    h, w, _ = image.shape
    # Dùng numpy array trực tiếp thay vì loop python để nhanh hơn
    landmark_array = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    h, w, _ = image.shape
    landmark_point = [[min(int(l.x * w), w - 1), min(int(l.y * h), h - 1)] for l in landmarks.landmark]
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    # Vector hóa việc trừ toạ độ sẽ nhanh hơn loop, nhưng list ngắn nên để loop cũng OK
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image_width, image_height, point_history):
    # Nhận width/height trực tiếp thay vì cả image object
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]

    for i in range(len(temp_point_history)):
        temp_point_history[i][0] = (temp_point_history[i][0] - base_x) / image_width
        temp_point_history[i][1] = (temp_point_history[i][1] - base_y) / image_height

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0: pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            csv.writer(f).writerow([number, *point_history_list])

# Tách hàm vẽ Menu ra cho gọn
def draw_menu_status(image, control_enabled):
    menu_color = (0, 255, 0) if control_enabled else (0, 0, 255)
    display_msg = "CONTROL: ON" if control_enabled else "CONTROL: OFF"
    cv.rectangle(image, (5, 65), (280, 110), (0, 0, 0), -1)
    cv.putText(image, display_msg, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, menu_color, 2, cv.LINE_AA)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Danh sách các cặp điểm kết nối (Định nghĩa 1 lần)
        connections = [
            (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12),
            (13,14), (14,15), (15,16), (17,18), (18,19), (19,20),
            (0,1), (1,2), (2,5), (5,9), (9,13), (13,17), (17,0)
        ]
        
        # Vẽ line
        for p1, p2 in connections:
            cv.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (255, 255, 255), 2)

        # Vẽ keypoints
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # Cổ tay
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            # Các điểm đầu ngón tay (vẽ to hơn)
            elif index in [4, 8, 12, 16, 20]:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            else:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    # Chỉ vẽ Finger Gesture nếu có text
    if finger_gesture_text != "":
        cv.putText(image, "Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

# Cần import copy ở global scope (đã có trong phần đầu file nhưng nhắc lại để chắc chắn)
import copy 

if __name__ == '__main__':
    main()