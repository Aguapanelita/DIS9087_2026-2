import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- CONFIGURACIÓN DE PANTALLA ---
ANCHO = 1280
ALTO = 720
TIEMPO_VIDA_LINEA = 2.0  # Duración del dibujo antes de desaparecer

# --- ESTRUCTURAS DE DATOS ---
puntos_dibujo = []
particulas = []

# --- CONFIGURACIÓN DE MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)

print("¡Efecto de Dedo Índice Iniciado! Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    t_actual = time.time()
    
    # Lienzo para efectos visuales
    lienzo_efectos = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Color base unificado (Celeste Neón Perfecto: BGR)
    COLOR_BASE_B = 255
    COLOR_BASE_G = 230
    COLOR_BASE_R = 100
    
    # 1. RASTREO DEL DEDO ÍNDICE
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_indice = hand_landmarks.landmark[8] # <--- Selecciona la punta del índice de forma correcta
            ix, iy = int(lm_indice.x * ANCHO), int(lm_indice.y * ALTO)
            
            if 0 <= ix < ANCHO and 0 <= iy < ALTO:
                puntos_dibujo.append({'pt': (ix, iy), 'time': t_actual})
                
                # Generar partículas exactas con el color heredado
                for _ in range(4):
                    particulas.append({
                        'pt': [float(ix), float(iy)],
                        'vel': [random.uniform(-5, 5), random.uniform(-5, 5)],
                        'time': t_actual,
                        'color': (COLOR_BASE_B, COLOR_BASE_G, COLOR_BASE_R) 
                    })

    # 2. ACTUALIZAR Y DIBUJAR LA LÍNEA BRILLANTE
    puntos_dibujo = [p for p in puntos_dibujo if t_actual - p['time'] < TIEMPO_VIDA_LINEA]
    
    for i in range(1, len(puntos_dibujo)):
        # Interpolación por cortes de pérdida de MediaPipe
        if puntos_dibujo[i]['time'] - puntos_dibujo[i-1]['time'] > 0.2:
            continue
            
        edad = t_actual - puntos_dibujo[i]['time']
        porcentaje_vida = max(0.0, 1.0 - (edad / TIEMPO_VIDA_LINEA))
        
        grosor = int(7 * porcentaje_vida) + 1
        
        b_linea = int(COLOR_BASE_B * porcentaje_vida)
        g_linea = int(COLOR_BASE_G * porcentaje_vida)
        r_linea = int(COLOR_BASE_R * porcentaje_vida)
        
        if grosor > 0:
            cv2.line(lienzo_efectos, puntos_dibujo[i-1]['pt'], puntos_dibujo[i]['pt'], (b_linea, g_linea, r_linea), grosor)

    # 3. ACTUALIZAR Y DIBUJAR LAS PARTÍCULAS
    particulas_vivas = []
    for p in particulas:
        edad = t_actual - p['time']
        if edad < TIEMPO_VIDA_LINEA:
            porcentaje_vida = max(0.0, 1.0 - (edad / TIEMPO_VIDA_LINEA))
            
            # Movimiento físico
            p['pt'][0] += p['vel'][0]
            p['pt'][1] += p['vel'][1]
            
            # Extraemos cada canal de la tupla de color individualmente
            b_part = int(p['color'][0] * porcentaje_vida)
            g_part = int(p['color'][1] * porcentaje_vida)
            r_part = int(p['color'][2] * porcentaje_vida)
            
            radio = int(5 * porcentaje_vida) + 1
            
            cv2.circle(lienzo_efectos, (int(p['pt'][0]), int(p['pt'][1])), radio, (b_part, g_part, r_part), -1)
            particulas_vivas.append(p)
            
    particulas = particulas_vivas

    # 4. FUSIÓN VISUAL SOBRE LA CÁMARA
    mascara_efectos = cv2.cvtColor(lienzo_efectos, cv2.COLOR_BGR2GRAY)
    _, mascara_binaria = cv2.threshold(mascara_efectos, 5, 255, cv2.THRESH_BINARY)
    
    fondo_camara = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mascara_binaria))
    resultado_final = cv2.add(fondo_camara, lienzo_efectos)

    cv2.imshow('Estela Magica de Particulas', resultado_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()