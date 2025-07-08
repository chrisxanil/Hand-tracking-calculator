import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# States
state = "first_number"
first_number = ""
second_number = ""
operator = ""
cooldown = 0
result = ""
last_eval_time = 0
transition_start_time = None
in_transition = False
prev_digit = None
digit_cooldown = 0

def count_fingers(hand):
    fingers = []
    if hand[4].x < hand[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    tips = [8, 12, 16, 20]
    base = [6, 10, 14, 18]
    for t, b in zip(tips, base):
        if hand[t].y < hand[b].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

def detect_zero(hand):
    dist = np.hypot(hand[8].x - hand[4].x, hand[8].y - hand[4].y)
    return dist < 0.05

def detect_thumbs_down(hand):
    thumb_tip_y = hand[4].y
    thumb_ip_y = hand[3].y
    other_fingers = [8, 12, 16, 20]
    folded = all(hand[tip].y > hand[tip - 2].y for tip in other_fingers)
    return folded and (thumb_tip_y > thumb_ip_y)

def detect_addition(h1, h2):
    dist = np.hypot(h1[8].x - h2[8].x, h1[8].y - h2[8].y)
    return dist < 0.07

def detect_division(hand):
    fingers = [False] * 5
    tips = [4, 8, 12, 16, 20]
    base = [3, 6, 10, 14, 18]
    for i in range(5):
        if i == 0:  # Thumb
            fingers[i] = hand[4].x < hand[3].x
        else:
            fingers[i] = hand[tips[i]].y < hand[base[i]].y
    return fingers == [True, True, False, False, True]

def detect_subtraction(hand):
    thumb_tip_y = hand[4].y
    thumb_ip_y = hand[3].y
    folded = all(hand[tip].y > hand[tip - 2].y for tip in [8, 12, 16, 20])
    return folded and (thumb_tip_y > thumb_ip_y)

def detect_multiplication(hand):
    tips = [4, 8, 12, 16, 20]
    base = [3, 6, 10, 14, 18]
    fingers = []
    for t, b in zip(tips, base):
        if t == 4:  # Thumb
            fingers.append(hand[t].x < hand[b].x)
        else:
            fingers.append(hand[t].y < hand[b].y)
    return fingers == [True, False, False, False, False]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    cooldown = max(0, cooldown - 1)
    digit_cooldown = max(0, digit_cooldown - 1)

    if results.multi_hand_landmarks:
        hands_list = results.multi_hand_landmarks
        hand1 = hands_list[0].landmark
        for hand in hands_list:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Digit input
        if state in ["first_number", "second_number"] and digit_cooldown == 0 and not in_transition:
            digit = None
            if detect_zero(hand1):
                digit = "0"
            else:
                fingers = count_fingers(hand1)
                if 1 <= fingers <= 5:
                    digit = str(fingers)
            if digit and digit != prev_digit:
                if state == "first_number":
                    first_number += digit
                else:
                    second_number += digit
                prev_digit = digit
                digit_cooldown = 25

        if count_fingers(hand1) == 0:
            prev_digit = None

        # Transition signal
        if detect_thumbs_down(hand1) and cooldown == 0 and not in_transition:
            if state == "first_number" and first_number:
                in_transition = True
                transition_start_time = time.time()
            elif state == "second_number" and second_number:
                in_transition = True
                transition_start_time = time.time()

        # Perform transition after 2s
        if in_transition and time.time() - transition_start_time >= 2:
            if state == "first_number":
                state = "operator"
            elif state == "second_number":
                try:
                    exp = f"{first_number}{operator}{second_number}"
                    result = f"{exp} = {eval(exp)}"
                except:
                    result = "Error"
                state = "done"
                last_eval_time = time.time()
            cooldown = 20
            in_transition = False

        # Operator detection with strict addition check
        if state == "operator" and cooldown == 0:
            operator = ""
            if len(hands_list) == 2:
                hand2 = hands_list[1].landmark
                if detect_addition(hand1, hand2):
                    operator = "+"
            if detect_multiplication(hand1):
                operator = "*"
            elif detect_division(hand1):
                operator = "/"
            elif detect_subtraction(hand1):
                operator = "-"
            if operator:
                state = "second_number"
                cooldown = 20

    if state == "done" and time.time() - last_eval_time > 5:
        state = "first_number"
        first_number = ""
        second_number = ""
        operator = ""
        result = ""
        prev_digit = None
        digit_cooldown = 0

    # UI
    cv2.putText(frame, f"State: {state}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Input: {first_number} {operator} {second_number}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    if result:
        cv2.putText(frame, f"Result: {result}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if in_transition:
        cv2.putText(frame, f"Waiting...", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3)

    cv2.imshow("Gesture Calculator", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
