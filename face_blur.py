import cv2

# Load the Haar Cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, img = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=7)

    # Create a copy of the image for drawing rectangles around faces
    detection = img.copy()

    # Loop through detected faces and apply a blur effect to censor them
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(detection, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Apply a blur effect to the region containing the face
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))

    # Display the original image with censored faces
    cv2.imshow("Censored", img)

    # Display the image with face detection rectangles
    cv2.imshow("Face Detection", detection)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
