import cv2

#Our Image
img_file='CarImage3.jpg'

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create opencv image
img = cv2.imread(img_file)

#convert to grayscale (needed for haarcascade)
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file) #brain

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)  #multiscale means detect cars of any scale


# print(cars)

# output [top,left,width,height]
# [[178 720  54  54]
#  [814 294  73  73]
#  [881 759  94  94]
#  [190 733  34  34]]
#[x,y,w,h]


# Draw rectangles around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,40,255),2)








# Display the image with the cars spotted
cv2.imshow('Car_detector',img) #pops a window with the image 

#Don't autoclose
cv2.waitKey() # waits for a key to be pressed to close window




print("cc")