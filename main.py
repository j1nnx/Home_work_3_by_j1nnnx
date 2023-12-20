import cv2
import mediapipe as mp
import mouse

cap = cv2.VideoCapture(0) #Камера
hands = mp.solutions.hands.Hands(max_num_hands=2, static_image_mode=1, min_tracking_confidence=0.5, min_detection_confidence=0.1) #Объект ИИ для определения ладони
draw = mp.solutions.drawing_utils #Для рисование ладони

mouse.move(1920, 1080)

class handTracker():
    def init(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])

        return lmlist

tracker = handTracker()


while True:
    #Закрытие окна
    if cv2.waitKey(1) & 0xFF == 27:
        break

    success, image = cap.read() #Считываем изображение с камеры



    image = tracker.handsFinder(image)
    lmList = tracker.positionFinder(image)

    if len(lmList) != 0:
        if ((lmList[8][2] - lmList[5][2]) < 10):
            mouse.click()

    image = cv2.flip(image, 1) #Отражаем изображение для корекктной картинки
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Конвертируем в rgb
    results = hands.process(imageRGB) #Работа mediapipe

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if (id == 8):
                    mouse.move(cx * 1920 / w, cy * 1080 / h)



    print(hands.process(image))







    cv2.imshow("Hand", image) #Отображаем картинку
