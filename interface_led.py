# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import numpy as np
import threading

# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import numpy as np
import threading

# Import RPi.GPIO
import RPi.GPIO as GPIO
import time


# Define the L1Dist custom layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Build app and layout
class CamApp(App):
    
    def build(self):
        # GPIO setup
        GPIO.cleanup()  # Add this line to reset GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Define GPIO pins
        self.SENSOR_PIN = 27
        self.LED_PIN_GREEN = 17  # Pin for green LED
        self.LED_PIN_RED = 26    # Pin for red LED

        # Set GPIO pin modes
        GPIO.setup(self.SENSOR_PIN, GPIO.IN)
        GPIO.setup(self.LED_PIN_GREEN, GPIO.OUT)
        GPIO.setup(self.LED_PIN_RED, GPIO.OUT)

        # Turn off LEDs initially
        GPIO.output(self.LED_PIN_GREEN, GPIO.LOW)
        GPIO.output(self.LED_PIN_RED, GPIO.LOW)

        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .2))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        try:
            self.model = tf.keras.models.load_model('/home/pi/Desktop/licenta/siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        # Start the thread to check for obstacles
        threading.Thread(target=self.check_for_obstacle, daemon=True).start()

        return layout

    def check_for_obstacle(self):
        try:
            while True:
                if GPIO.input(self.SENSOR_PIN):
                    # print("Niciun obstacol.")
                    pass
                else:
                    # print("Obstacol detectat!")
                    self.verify()  # Pornește verificarea
                time.sleep(1)
        except KeyboardInterrupt:
            print("Programul a fost oprit de utilizator.")
        finally:
            GPIO.cleanup()

    # Run continuously to get webcam feed
    def update(self, *args):
            # Read frame from opencv
            ret, frame = self.capture.read()
            if not ret or frame is None:
                print("Failed to capture image from camera")
                return
        
            # Crop the frame (make sure the cropping indices are within the frame's dimensions)
            frame = frame[120:120 + 250, 200:200 + 250, :]
        
            # Flip horizontally and convert image to texture
            buf = cv2.flip(frame, -1).tostring()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture


    # Load image from file and convert to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)

        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        # Scale image to be between 0 and 1
        img = img / 255.0

        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Debugging: Print out the shape of the saved frame
        print(f"Frame shape: {frame.shape}")

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Debugging: Print out the shapes of the preprocessed images
            print(f"Input image shape: {input_img.shape}")
            print(f"Validation image shape: {validation_img.shape}")

            # Make Predictions
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

            # Debugging: Print out the prediction result
            print(f"Result for {image}: {result}")

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # Set verification textS
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details
        Logger.info(f'Results: {results}')
        Logger.info(f'Detection: {detection}')
        Logger.info(f'Verification: {verification}')
        Logger.info(f'Verified: {verified}')

        # Debugging: Print out the final verification status
        print(f"Final verification: {verified}")

        # Update LED status
        GPIO.output(self.LED_PIN_GREEN, GPIO.HIGH if verified else GPIO.LOW)
        GPIO.output(self.LED_PIN_RED, GPIO.LOW if verified else GPIO.HIGH)

        # Turn off LEDs after 5 seconds
        threading.Timer(5, self.turn_off_leds).start()

        return results, verified

    def turn_off_leds(self):
        GPIO.output(self.LED_PIN_GREEN, GPIO.LOW)
        GPIO.output(self.LED_PIN_RED, GPIO.LOW)

if __name__ == '__main__':
    try:
        CamApp().run()
    except KeyboardInterrupt:
        print("Cleaning up GPIO...")
        GPIO.cleanup()
