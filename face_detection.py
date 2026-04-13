import cv2

# 1. Pre-trained Face Detection model load karna
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Webcam start karna
cap = cv2.VideoCapture(0)

print("Webcam start ho raha hai... Band karne ke liye 'q' dabayein.")

while True:
    # Frame-by-frame capture karna
    ret, frame = cap.read()
    
    # Image ko grayscale mein badalna (detection ke liye zaroori hai)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Chehre detect karna
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Har chehre ke charon taraf rectangle banana
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Result dikhana
    cv2.imshow('Face Detection System', frame)

    # 'q' dabane par loop tod dena
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sab kuch band karna
cap.release()
cv2.destroyAllWindows()