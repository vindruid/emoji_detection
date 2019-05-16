import cv2
from keras.models import load_model
import numpy as np
import os
from random import randint
import time

model = load_model('RPS.h5')


def calcResult(pred_class, cpu):
    if pred_class == cpu:
        return 'draw'
    if pred_class == 1 and (cpu == 3 or cpu == 4):
        return 'user'
    if pred_class == 2 and (cpu == 1 or cpu == 5):
        return 'user'
    if pred_class == 3 and (cpu == 2 or cpu == 4):
        return 'user'
    if pred_class == 4 and (cpu == 2 or cpu == 5):
        return 'user'
    if pred_class == 5 and (cpu == 1 or cpu == 3):
        return 'user'
    return 'cpu'


def main():
    flag = 0
    result = ''
    emojis = get_emojis()
    img_rules = cv2.imread('rules.png', -1)
    img_rules = cv2.resize(img_rules, (300, 300))
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    #Initial focused window
    x, y, w, h = 300, 50, 350, 350

    play = False


    while (cap.isOpened()):
        # Set timer 
        if not play: 
            start_time = time.time()

        ret, img_origin = cap.read()
        img_origin = cv2.flip(img_origin, 1)
        if not play:
            
            img = img_origin.copy()
            gray_layer = img_origin.copy()
            img_h, img_w = img.shape[0], img.shape[1]
            cv2.rectangle(gray_layer, (0, 0),
                          (img_w, img_h), (200, 200, 200), -1)

            alpha = 0.5
            cv2.addWeighted(gray_layer, alpha, img, 1 - alpha, 0, img)

            img_rules_x = int(img_w / 2) - 150
            img_rules_y = int(img_h / 4)
            img[img_rules_y : img_rules_y + 300, img_rules_x : img_rules_x + 300, :] = img_rules 

            text_x = int(img_w/ 6)
            text_y = int(img_h/ 6)

            cv2.putText(img, 'Press SPACE to Play', (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 2)

            cv2.imshow("Frame", img)
            k = cv2.waitKey(10)
            if k == 27: #ESC to quit
                break

            if k == 32: #Space to start Game
                play = True
                play_set = 0
                play_set_past = 0
                score_player = 0
                score_cpu = 0
                score_changeable = True
                cpu = (randint(1, 5))

        if play: 

            
            if play_set != play_set_past: # when new set
                start_time = time.time()
                play_set_past = play_set
                score_changeable = True
                cpu = (randint(1, 5))

            now_time = time.time()

            # count until 3 second to start
            if now_time - start_time < 3:
                img = img_origin.copy()
                if now_time - start_time < 1:
                    cv2.putText(img, '1', (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 250, 0), 2)
                elif now_time - start_time < 2:
                    cv2.putText(img, '1 2', (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 250, 0), 2)
                elif now_time - start_time < 3:    
                    cv2.putText(img, '1 2 3', (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 250, 0), 2)      

                cv2.putText(img, "Player Score: " + str(score_player) , (text_x, text_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)  
                cv2.putText(img, "CPU Score: " + str(score_cpu) , (text_x, text_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)  
                cv2.imshow("Frame", img)

                k = cv2.waitKey(10)
                if k == 27:
                    play = False
            
            elif now_time - start_time < 5: #Show time 061024
                img = img_origin.copy()
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
                res = cv2.bitwise_and(img, img, mask=mask2)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                median = cv2.GaussianBlur(gray, (5, 5), 0)

                kernel_square = np.ones((5, 5), np.uint8)
                dilation = cv2.dilate(median, kernel_square, iterations=2)
                opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
                ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

                thresh = thresh[y:y + h, x:x + w]
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # When countours detected
                # Predict the countour and randomized CPU
                if len(contours) > 0:
                    contour = max(contours, key=cv2.contourArea)
                    # when countour size large enough
                    if cv2.contourArea(contour) > 2500:
                        # if flag == 0:
                        #     cpu = (randint(1, 5))
                        #     flag = 1
                        x, y, w1, h1 = cv2.boundingRect(contour)
                        newImage = thresh[y:y + h1, x:x + w1]
                        newImage = cv2.resize(newImage, (50, 50))
                        pred_probab, pred_class = keras_predict(model, newImage)
                        # print(pred_class, pred_probab)
                        img = overlay(img, emojis[pred_class], 370, 50, 90, 90)
                        img = overlay(img, emojis[cpu], 530, 50, 90, 90)
                        result = calcResult(pred_class, cpu)

                elif len(contours) == 0:
                    flag = 0
                x, y, w, h = 300, 50, 350, 350
                cv2.putText(img, 'USER', (380, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, 'CPU', (550, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, 'Winner : ', (420, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if result == 'user':
                    cv2.putText(img, 'USER', (530, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif result=='cpu':
                    cv2.putText(img, 'CPU', (530, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif result=='draw':
                    cv2.putText(img, 'DRAW', (530, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    pass
                cv2.imshow("Frame", img)
                cv2.imshow("Contours", thresh)
                k = cv2.waitKey(10)
                if k == 27:
                    play = False

            elif now_time - start_time < 7: #Show result
                img = img_origin.copy()
                img = overlay(img, emojis[pred_class], 370, 50, 90, 90)
                img = overlay(img, emojis[cpu], 530, 50, 90, 90)

                cv2.putText(img, 'USER', (380, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, 'CPU', (550, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(img, 'Winner : ', (100, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
                if result == 'user':
                    cv2.putText(img, 'USER', (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
                    if score_changeable:
                        score_player += 1
                        score_changeable = False
                elif result=='cpu':
                    cv2.putText(img, 'CPU', (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
                    if score_changeable:
                        score_cpu += 1
                        score_changeable = False
                elif result=='draw':
                    cv2.putText(img, 'DRAW', (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 2)

                cv2.putText(img, "Player Score: " + str(score_player) , (text_x, text_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)  
                cv2.putText(img, "CPU Score: " + str(score_cpu) , (text_x, text_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)  

                cv2.imshow("Frame", img)
                k = cv2.waitKey(10)
                if k == 27:
                    play = False

            else:
                play_set += 1



def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def get_emojis():
    emojis_folder = 'RPS_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder + str(emoji) + '.png', -1))
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()
