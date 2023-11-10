import numpy as np
import time
from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
import socket
import select

SERVER_IP = "127.0.0.1"
SERVER_PORT = 9000
timeout_value = 600

font = cv2.FONT_HERSHEY_SIMPLEX
classNames = ["Empty_Slot", "Filled_Slot"]
model = YOLO(r"C:\Users\Rednax\PycharmProjects\GKN_Project\runs\detect\train\weights\best.pt")

def parse_data(data):
    parts = data.split(',')
    if len(parts) != 3:
        print("Error: Received unexpected data format from server:", data)
        return None, None, None
    try:
        signal = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
    except ValueError:
        print("Error: Received data from server could not be converted to integers:", data)
        return None, None, None
    return signal, x, y
def create_socket(ip, port, timeout):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(timeout)
    client_socket.setblocking(0)  # set the socket to non-blocking
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)  # disable Nagle's algorithm

    # try to establish connection
    try:
        client_socket.connect((ip, port))
    except BlockingIOError:
        # if connection cannot be established immediately, wait until it's ready
        ready = select.select([], [client_socket], [], timeout)
        if ready[1]:
            print("Connection established")
        else:
            print("Unable to establish connection within timeout period")
            client_socket.close()
            return None

    # wait for data to be available
    ready = select.select([client_socket], [], [], timeout)
    if ready[0]:
        response = client_socket.recv(1024)
        print("Received from server:", response.decode())
    else:
        print("No response from server within timeout period")
        client_socket.close()
        return None

    return client_socket
def open_camera(device_id=1, frame_size=(1280, 720)):
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW )
    cap.set(3, frame_size[0])
    cap.set(4, frame_size[1])

    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        exit()
    return cap

def draw_roi(frame, result):
    boxes = result.boxes
    detected_class = None
    roi = None
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h),l= 9)
        conf = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        detected_class = classNames[cls]
        cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6,
                           thickness=1, offset=3)
        roi = (x1, y1, w, h)
        print(x1, x2, w, h)
    if roi is None and detected_class is None:
        return None, None
    return roi, detected_class


def send_socket_data(socket, frame, roi, detected_class):
    if detected_class not in ["Empty_Slot", "Filled_Slot"]:
        return
    cropped_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    gray_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(gray_cropped_frame)
    if (M["m00"] == 0): M["m00"] = 1
    x = int(M["m10"] / M["m00"]) + roi[0]
    y = int(M["m01"] / M["m00"]) + roi[1]
    cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
    cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
    x = max(0, min(x, 1280))  # adjust the maximum value according to your frame size
    y = max(0, min(y, 720))
    signal = 1 if detected_class == 'Filled_Slot' else 0
    if signal not in [0, 1]:
        print("Error: Invalid signal value")
        return
    socket.send((str(signal) + ',' + str(x) + ',' + str(y)).encode())
    time.sleep(0.2)
    print(f"The X-value coordinate: {x}")
    print(f"The Y-value coordinate:{y}")
    print(f"Signal: {signal}")

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def main():
    client_socket = create_socket(SERVER_IP, SERVER_PORT, timeout_value)
    cap = open_camera()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from the camera.")
            break
        gamma_corrected_frame = adjust_gamma(frame, gamma=2.0)

        # Split the channels
        b, g, r = cv2.split(gamma_corrected_frame)

        # Apply Histogram Equalization to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b_histeq = clahe.apply(b)
        g_histeq = clahe.apply(g)
        r_histeq = clahe.apply(r)

        # Merge the channels back together
        histeq = cv2.merge([b_histeq, g_histeq, r_histeq])

        results = model(histeq, stream=True)
        detected_class = None
        conf = 0
        roi = None

        for r in results:
            roi, detected_class = draw_roi(frame, r)

        if ret == True:
            if detected_class == "Filled_Slot" or detected_class == "Empty_Slot":
                frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                send_socket_data(client_socket, frame, roi, detected_class)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()


if __name__ == "__main__":
    main()