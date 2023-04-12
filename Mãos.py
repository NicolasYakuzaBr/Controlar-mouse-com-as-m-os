import cv2
import mediapipe as mp
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Definindo os parâmetros para o mediapipe Hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Configurando a tela do computador
screen_size = pyautogui.size()

# Definindo o tamanho da janela do OpenCV
WINDOW = "Captura de Gestos"
cv2.namedWindow(WINDOW)
cv2.moveWindow(WINDOW, 0, 0)

# Inicializando a câmera
cap = cv2.VideoCapture(0)

# Variável para controlar o tempo entre os cliques
last_click_time = 0

# Variáveis para manter a posição do cursor
last_x = None
last_y = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convertendo a imagem para RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Obtendo os resultados do mediapipe Hands
    results = hands.process(image)

    # Desenhando as landmarks das mãos na imagem
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtendo as coordenadas (x, y) da ponta do dedo indicador da mão direita
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger.x * image.shape[1])
            y = int(index_finger.y * image.shape[0])

            # Movendo o cursor do mouse para as coordenadas obtidas
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                if last_x is None or last_y is None:
                    pyautogui.moveTo(x, y)
                else:
                    pyautogui.moveTo(last_x, last_y)

                # Atualizando a posição anterior do cursor
                last_x = x
                last_y = y

                # Verificando se o tempo entre os cliques é maior que 2 segundos
                current_time = time.time()
                if current_time - last_click_time > 2:
                    pyautogui.click(x, y)
                    last_click_time = current_time

    # Mostrando a imagem na janela do OpenCV
    cv2.imshow(WINDOW, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Encerrando o programa ao pressionar a tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberando os recursos
cap.release()
cv2.destroyAllWindows()
