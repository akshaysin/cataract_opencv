import cv2
import sys

imagePath = sys.argv[1]
cascPath = "cascade.xml"

pedsCascade =  cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
# resized_img = cv2.resize(image, (128, 128))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect coins in pic

catarat = pedsCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=2,
        minSize=(50, 50)
)

print("Found {0} catarats!".format(len(catarat)))

# Draw a rectangle around the peds
for (x, y, w, h) in catarat:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("Faces found", image)
status = cv2.imwrite('catarat_saved.jpg', image)
print ("Image written to file-system : ",status)
