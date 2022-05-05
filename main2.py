import cv2

from textAreaDetector2 import LicencePlatesParser2
import config


def run():
    debug = False
    parser = LicencePlatesParser2(debug)
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_counter % config.PARSE_EVERY_N_FRAME == 0:
            text_result, debug_image = parser.run(frame)
            if text_result:
                print(text_result)
            if debug and debug_image is not None:
                frame = debug_image.copy()
                cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_counter += 1


run()
