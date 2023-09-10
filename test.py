import cv2

# Haar-cascade sınıflandırıcıları
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Kamera bağlantısını başlatma
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare okuma
    ret, frame = cap.read()
    
    # Gri tonlamaya dönüştürme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit etme
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Bulunan yüzlerin etrafına dikdörtgen çizme
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Yüz bölgesini gri tonlamaya dönüştürme
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Gözleri tespit etme
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            # Bulunan gözlerin etrafına dikdörtgen çizme
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Sonuçları gösterme
    cv2.imshow('Face and Eye Tracking', frame)
    
    # 'q' tuşuna basıldığında döngüyü sonlandırma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
