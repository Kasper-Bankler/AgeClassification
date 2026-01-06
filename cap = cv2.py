cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)  # Sandsynligheder
                pred_class = torch.argmax(probs, dim=1).item()
                label = class_names[pred_class]

            # --- Tegn rektangel og label ---
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            # Vis prediktion + sandsynligheder
            prob_text = ", ".join([f"{c}:{probs[0,i]:.2f}" for i,c in enumerate(class_names)])
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.imshow("Age Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()