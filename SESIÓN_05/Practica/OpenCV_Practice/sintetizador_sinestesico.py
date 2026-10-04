import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# Variables globales para conectar el video con el oscilador de audio
frecuencia_actual = 440.0
sonando = True

# --- MOTOR DE AUDIO ESTABLE (Generador de bloques de onda infinita) ---
def hilo_generador_audio():
    global frecuencia_actual, sonando
    sample_rate = 44100
    duracion_bloque = 0.1  # Bloques cortos de 100 milisegundos para continuidad
    
    while sonando:
        # Asegurar un rango audible seguro y cómodo
        freq = max(150.0, min(frecuencia_actual, 1000.0))
        
        # Crear la matriz de tiempo para este bloque específico
        t = np.linspace(0, duracion_bloque, int(sample_rate * duracion_bloque), False)
        
        # Generar una onda sinusoidal pura limpia
        # El 0.2 controla el volumen para que sea fuerte y claro
        onda = 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Reproducir el bloque de sonido de forma inmediata
        sd.play(onda, sample_rate)
        
        # Esperar exactamente el tiempo del bloque para encadenar el siguiente sin baches
        time.sleep(duracion_bloque - 0.01)

# Lanzar el motor de audio estable en segundo plano
hilo = threading.Thread(target=hilo_generador_audio, daemon=True)
hilo.start()

# Cargar el detector de rostros nativo de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Sintetizador Sinestésico Permanente Activo.")
print("Coloca cualquier objeto Amarillo, Azul o Rojo frente a tu pecho.")
print("Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    output_frame = frame.copy()

    # Pasar a escala de grises para el detector de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(rostros) > 0:
        for (fx, fy, fw, fh) in rostros:
            # Encuadre dinámico del torso (zona de la playera)
            x1 = max(0, fx - int(fw * 0.3))
            y1 = max(0, fy)
            x2 = min(width, fx + fw + int(fw * 0.3))
            y2 = min(height, fy + fh + int(fh * 2.2))

            zona_persona = frame[y1:y2, x1:x2]

            if zona_persona.size > 0:
                # Extraer promedios RGB de esa región
                promedios = cv2.mean(zona_persona)
                b_prom = promedios[0] # Azul
                g_prom = promedios[1] # Verde
                r_prom = promedios[2] # Rojo

                # MATEMÁTICA SINESTÉSICA:
                # El azul sube los Hz (tonos agudos), el rojo los baja (tonos graves)
                frecuencia_actual = 250.0 + (b_prom * 3.5) - (r_prom * 1.8) + (g_prom * 1.0)

                # Dibujar el indicador visual redondo arriba a la izquierda
                cv2.circle(output_frame, (30, 30), 15, (int(b_prom), int(g_prom), int(r_prom)), -1)
                cv2.circle(output_frame, (30, 30), 15, (255, 255, 255), 2)

    # Mostrar la ventana de video fluida
    cv2.imshow("Sintetizador Sinestesico Continuo", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        sonando = False
        break

# Detener el audio de golpe al salir
sd.stop()
cap.release()
cv2.destroyAllWindows()