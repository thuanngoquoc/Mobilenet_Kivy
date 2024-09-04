import kivy
kivy.require('2.0.0')  # Xác định phiên bản Kivy

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import tensorflow as tf

class CameraApp(App):

    def build(self):
        # Mở camera
        self.capture = cv2.VideoCapture(0)

        # Tải mô hình TensorFlow Lite
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()

        # Tạo giao diện
        self.frame = Image()
        self.button = Button(text="Detect")
        self.button.bind(on_press=self.detect_objects)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.frame)
        layout.add_widget(self.button)

        # Lên lịch cập nhật khung hình từ camera
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Chuyển đổi khung hình từ BGR sang RGB và thay đổi kích thước
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_frame = cv2.resize(input_frame, (300, 300))

            # Chuẩn bị dữ liệu đầu vào cho mô hình
            input_data = np.expand_dims(input_frame, axis=0).astype(np.float32)
            input_data = (input_data / 255.0).astype(np.float32)

            # Chạy mô hình nhận diện
            input_details = self.model.get_input_details()[0]
            output_details = self.model.get_output_details()[0]
            self.model.set_tensor(input_details['index'], input_data)
            self.model.invoke()
            boxes = self.model.get_tensor(output_details['index'])[0]  # Lấy kết quả nhận diện

            # Vẽ hình chữ nhật lên các đối tượng được phát hiện
            for box in boxes:
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * frame.shape[0])
                xmin = int(xmin * frame.shape[1])
                ymax = int(ymax * frame.shape[0])
                xmax = int(xmax * frame.shape[1])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Chuyển đổi khung hình sang texture để hiển thị trong Kivy
            buf = frame.tostring()
            img = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img.blit_buffer(buf, bufferfmt='ubyte')
            self.frame.texture = img

    def detect_objects(self, instance):
        # Chức năng này có thể dùng để kích hoạt nhận diện thủ công
        pass

if __name__ == '__main__':
    CameraApp().run()
 