import cv2

#Our Image
img_file='CarImage.jpg'

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create opencv image
img = cv2.imread(img_file)

#convert to grayscale (needed for haarcascade)
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file) #brain







# Display the image with the cars spotted
cv2.imshow('Car_detector',black_n_white) #pops a window with the image 

#Don't autoclose
cv2.waitKey() # waits for a key to be pressed to close window




print("cc")