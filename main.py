import cv2
import tkinter as tk
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, video_source=0):
        self.root = tk.Tk()
        self.root.title("Face Recognition")

        self.video_source = video_source

        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(self.root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_capture = tk.Button(self.root, text="Capture", command=self.capture_image)
        self.btn_capture.pack()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.delay = 10
        self.update()

        self.root.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(self.delay, self.update)

    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("captured_image.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    FaceRecognitionApp()


