import RPi.GPIO as GPIO
import time
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import numpy as np
from kivy.logger import Logger


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
        GPIO.cleanup()  # Reset GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Define GPIO pins
        self.SENSOR_PIN = 27
        self.BUTTON_PIN = 6  # Pin pentru buton
        self.LED_PIN_GREEN = 17  # Pin pentru LED verde
        self.LED_PIN_RED = 26    # Pin pentru LED roșu

        # Set GPIO pin modes
        GPIO.setup(self.SENSOR_PIN, GPIO.IN)
        GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.LED_PIN_GREEN, GPIO.OUT)
        GPIO.setup(self.LED_PIN_RED, GPIO.OUT)

        # Oprește LED-urile inițial
        GPIO.output(self.LED_PIN_GREEN, GPIO.LOW)
        GPIO.output(self.LED_PIN_RED, GPIO.LOW)

        # Componentele principale ale interfeței
        self.title = "Smile :)"
        self.web_cam = Image(size_hint=(1, .8))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .2))

        # Adaugă elemente în layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)

        # Încarcă modelul TensorFlow/Keras
        try:
            self.model = tf.keras.models.load_model('/home/pi/Desktop/licenta/siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        # Pornește thread-urile pentru verificarea obstacolelor și a butonului
        threading.Thread(target=self.check_for_obstacle, daemon=True).start()
        threading.Thread(target=self.check_for_button_press, daemon=True).start()

        return layout

    def check_for_obstacle(self):
        try:
            while True:
                if GPIO.input(self.SENSOR_PIN):
                    pass
                else:
                    self.verify()  # Pornește verificarea
                time.sleep(1)
        except KeyboardInterrupt:
            print("Program stopped by user.")
        finally:
            GPIO.cleanup()

    def check_for_button_press(self):
        try:
            while True:
                if GPIO.input(self.BUTTON_PIN) == GPIO.HIGH:
                    Clock.schedule_once(lambda dt: self.open_name_popup())
                time.sleep(1)
        except KeyboardInterrupt:
            print("Program stopped by user.")
        finally:
            GPIO.cleanup()

    def open_name_popup(self):
        layout = BoxLayout(orientation='vertical')
        self.name_input = TextInput(hint_text="Introdu numele persoanei", multiline=False)
        btn_submit = Button(text="Submit", on_press=self.capture_images)

        layout.add_widget(self.name_input)
        layout.add_widget(btn_submit)

        self.popup = Popup(title="Introdu numele persoanei", content=layout, size_hint=(None, None), size=(400, 200))
        self.popup.open()

    def capture_images(self, instance):
        name = self.name_input.text
        if not name:
            self.verification_label.text = "Numele nu a fost introdus."
            self.popup.dismiss()
            return

        # Creează directorul pentru persoana nouă
        person_dir = os.path.join('application_data', 'new_faces', name)
        os.makedirs(person_dir, exist_ok=True)
        
        images = []
        capture_duration = 10  # durata în secunde pentru capturarea imaginilor
        start_time = time.time()
        
        while time.time() - start_time < capture_duration:
            ret, frame = self.capture.read()
            if not ret or frame is None or frame.size == 0:
                print("Eroare la capturarea imaginii.")
                self.verification_label.text = "Eroare la capturarea imaginii."
                break

            img_path = os.path.join(person_dir, f"{len(images)}.jpg")
            cv2.imwrite(img_path, frame)  # Salvează imaginea
            images.append(self.preprocess_image(frame))
            print("Fotografia a fost salvată.")
            time.sleep(1)  # capturează o imagine la fiecare secundă
        
        if images:
            pairs, labels = self.generate_pairs_and_labels(images)
            self.finetune_model(pairs, labels)
            self.verification_label.text = "Modelul a fost antrenat cu succes."
        else:
            self.verification_label.text = "Nu s-au capturat imagini."

        self.popup.dismiss()

    def preprocess_image(self, image):
            if isinstance(image, str):
                image = cv2.imread(image)  
                if image is None:
                    print(f"Eroare la citirea imaginii de la calea: {image}")
                    return None
            elif isinstance(image, np.ndarray):
                pass  # Imaginea este deja un numpy array, deci nu mai este nevoie de citire
            else:
                print("Tipul obiectului nu este suportat pentru procesare.")
                return None

            image = cv2.resize(image, (100, 100))
            image = np.array(image, dtype=np.float32)
            image /= 255.0
            return image



    def generate_pairs_and_labels(self, images):
        pairs = []
        labels = []
        num_images = len(images)

        # Generate positive pairs (same person)
        for i in range(num_images):
            for j in range(i + 1, num_images):
                pairs.append([images[i], images[j]])
                labels.append(1)

        # Generate negative pairs (different people)
        for i in range(num_images):
            for j in range(num_images):
                if i != j:
                    pairs.append([images[i], images[j]])
                    labels.append(0)

        return np.array(pairs), np.array(labels)

    def finetune_model(self, pairs, labels):
        pairs = [np.array([pair[0] for pair in pairs]), np.array([pair[1] for pair in pairs])]
        pairs = [np.array(pairs[0]).reshape(-1, 100, 100, 3), np.array(pairs[1]).reshape(-1, 100, 100, 3)]
        model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(pairs, labels, epochs=15)
        
        # Save the trained model
        model.save('siamesemodel.h5')  # Save the trained model

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
            best_match_name = None
            best_match_score = 0

            # Iterate over all saved faces in new_faces directory
            for person_name in os.listdir('/home/pi/Desktop/licenta/app/application_data/new_faces'):
                person_dir = os.path.join('/home/pi/Desktop/licenta/app/application_data/new_faces', person_name)
                if not os.path.isdir(person_dir):  # Check if it's a directory
                    continue
                
                for image in os.listdir(person_dir):
                    input_img = self.preprocess_image(SAVE_PATH)
                    validation_img = self.preprocess_image(os.path.join(person_dir, image))

                    # Make Predictions
                    result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
                    results.append(result)

                    # Check for best match
                    if result > best_match_score:
                        best_match_score = result
                        best_match_name = person_name

                    # Debugging: Print out the prediction result
                    print(f"Result for {image}: {result}")

            # Detection Threshold: Metric above which a prediction is considered positive
            detection = np.sum(np.array(results) > detection_threshold)

            # Verification Threshold: Proportion of positive predictions / total positive samples
            verification = detection / len(results) if len(results) > 0 else 0
            verified = verification > verification_threshold

            # Set verification text
            if verified and best_match_name:
                self.verification_label.text = f'{best_match_name} Verified'
            else:
                self.verification_label.text = 'Unverified'

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
