# Import standard dependencies
import cv2
import os 
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

#Import uuid library to generate unique image names (universally unique identifiers)
import uuid

# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

from tensorflow.keras import backend as K
#Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
# O_PATH = os.path.join('data', 'o')

# #Make the directories
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# #Move  LFW Images to the following repository data/negative
# for directory in os.listdir('lfw'):
	# for file in os.listdir(os.path.join('lfw', directory)):
		# EX_PATH = os.path.join('lfw', directory, file)
		# NEW_PATH = os.path.join(NEG_PATH, file)
		# os.replace(EX_PATH, NEW_PATH)

#Establish  a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
	ret, frame = cap.read()
	
	#Cut down frame to 250x250px
	frame = frame[120:120+250, 200:200+250, :]
	#Mirror the image
	flipped = cv2.flip(frame, 1)
	
	#Collect anchors
	if cv2.waitKey(1) & 0XFF == ord('a'):
		#Create the unique file path
		imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
		#Write out anchor image
		cv2.imwrite(imgname, frame) 
	
	#Collect positives
	if cv2.waitKey(1) & 0XFF == ord('p'):
		#Create the unique file path
		imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
		#Write out anchor image
		cv2.imwrite(imgname, frame) 
	#Collect positives
	if cv2.waitKey(1) & 0XFF == ord('o'):
		#Create the unique file path
		imgname = os.path.join(O_PATH, '{}.jpg'.format(uuid.uuid1()))
		#Write out anchor image
		cv2.imwrite(imgname, frame) 
	
	cv2.imshow('Image Collector', flipped)
	if cv2.waitKey(1) & 0XFF == ord('q'):
		break

#Realease the webcam
cap.release()
#Close the image show frame
cv2.destroyAllWindows()

# cv2.imshow("Frame",frame[120:120+250,200:200+250, :])




anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

dir_test = anchor.as_numpy_iterator()

print(dir_test.next())

#Preprocessing - Scale & Resize
def preprocess(file_path):
	
	#Read in image from file path 
	byte_img = tf.io.read_file(file_path)
	#Load in the image
	img = tf.io.decode_jpeg(byte_img)
	#Preprocessing steps - resizing the image to be 100x100x3
	img = tf.image.resize(img, (100,100))
	#Scaling image to be between 0 and 1
	img = img / 255.0
	return img

	

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

exampple = samples.next()

print(exampple)

#Build Train and Test Partition
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*exampple)
# cv2.imshow("Build Train and Test Partition",res[1])

res = preprocess_twin(*exampple)

img_to_show = res[1].numpy()  # Convert the tensor in numpy array
img_to_show = (img_to_show * 255).astype(np.uint8)  # Convert to uint8

# Afisează imaginea
cv2.imshow("Build Train and Test Partition", img_to_show)
cv2.waitKey(0)  
cv2.destroyAllWindows()  


# res[2]

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

sample = data.as_numpy_iterator()
sam = sample.next()

# Show image
cv2.imshow("Sample", sam[1])
cv2.waitKey(0)  
cv2.destroyAllWindows()  
print(sam[2])
#print(round(len(data)*.7)) #420
# Training partition
train_data = data.take(round(len(data)*.7)) #first 420 images
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7)) #skip first 420 images
test_data = test_data.take(round(len(data)*.3)) #last 180 images
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

#Build Embedding Layer
inp = Input(shape=(100,100,3), name='input_image')
c1 = Conv2D(64, (10,10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)
mod = Model(inputs=[inp], outputs=[d1], name='embedding')
mod.summary()

def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(32, (5,5), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(64, (10,10), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=d1)
    
embedding = make_embedding()
embedding.summary()
input_test = np.random.random((1, 100, 100, 3))  # Crează un input de test
output = embedding.predict(input_test)
print(type(output))


with tf.GradientTape() as tape:
    inputs = tf.constant(input_test, dtype=tf.float32)
    tape.watch(inputs)
    outputs = embedding(inputs)

print("Outputs type:", type(outputs))
print("Outputs shape:", outputs.shape)
#Build Distance Layer
# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
        #print(input_embedding.shape, validation_embedding.shape)

# l1 = L1Dist()
# l1(anchor_embedding, validation_embedding)

#Make Siamese Model
input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)
siamese_layer = L1Dist()
distances = siamese_layer(inp_embedding, val_embedding)
classifier = Dense(1, activation='sigmoid')(distances)
classifier
siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
siamese_network.summary()

def make_siamese_model(embedding): 
    
    # Handle inputs   
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Embedding for every images
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(inp_embedding, val_embedding)
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

embedding_model = make_embedding()
siamese_model = make_siamese_model(embedding_model)
siamese_model.summary()

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
siamese_model.compile(optimizer=opt, loss=binary_cross_loss, metrics=['accuracy'])
checkpoint_dir = './training_checkpoints1'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
#Build train step function
test_batch = train_data.as_numpy_iterator()
batch_1 = test_batch.next()
X = batch_1[:2]
y = batch_1[2]
@tf.function #compiles a function into a tensorflow graph
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Return loss
    return loss
    
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx+1)
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 2
train(train_data, EPOCHS)

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
# Make predictions
y_hat = siamese_model.predict([test_input, test_val])
print(y_hat)
# Post processing the results 
[1 if prediction > 0.5 else 0 for prediction in y_hat ]
print(y_true)
#Calculate Metrics
# Creating a metric object 
m = Recall()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
print(m.result().numpy())

# Creating a metric object 
m = Precision()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
print(m.result().numpy())

#Viz Results
# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()

# Save weights
siamese_model.save('siamesemodel1.h5')

# Reload model 
model = tf.keras.models.load_model('siamesemodel1.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Make predictions with reloaded model
print(model.predict([test_input, test_val]))

# View model summary
model.summary()

#Verification in real time
#os.listdir(os.path.join('application_data', 'verification_images'))
#application_data/verification_images
#os.listdir(os.path.join('application_data', 'verification_images'))
#os.listdir(os.path.join('application_data', 'verification_images'))
for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    #print(validation_img)
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified
    
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    flipped = cv2.flip(frame, 1)
    cv2.imshow('Verification', flipped)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(model, 0.5, 0.5)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

np.sum(np.squeeze(results) > 0.9)
results

