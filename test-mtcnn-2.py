import cv2
from PIL import Image
import mtcnn

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    min_face_size = 20.0

    model = mtcnn.MTCNN()

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            image = Image.fromarray(frame[..., ::-1])

            bounding_boxes, _ = model.detect_faces(image)

            for i in range(len(bounding_boxes)):  # .shape[0]):
                bbox = bounding_boxes[i, :]
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 6)

            cv2.imshow('face Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()