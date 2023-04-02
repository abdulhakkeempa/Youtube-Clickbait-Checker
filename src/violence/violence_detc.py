import numpy as np
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import os
import time
from keras.models import load_model
from collections import deque
from pytube import YouTube
import sys

IMG_SIZE = 128
RESULT =[]

def print_results(video, limit=None)->str:
        # fig=plt.figure(figsize=(16, 30))
        # if not os.path.exists('output'):
        #     os.mkdir('output')

        print("Loading model ...")
        try:
          model = load_model('./model/model.h5')
        except Exception as e:
          print(e)
          print("Unable to load the model")
          sys.exit()

        Q = deque(maxlen=128)

        vs = cv2.VideoCapture(video)
        writer = None
        (W, H) = (None, None)
        count = 0     
        while True:
                (grabbed, frame) = vs.read()
                ID = vs.get(1)
                if not grabbed:
                    break
                try:
                    if (ID % 7 == 0):
                        count = count + 1
                        n_frames = len(frame)
                        
                        if W is None or H is None:
                            (H, W) = frame.shape[:2]

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        output = cv2.resize(frame, (512, 360)).copy()
                        frame = cv2.resize(frame, (128, 128)).astype("float16")
                        frame = frame.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                        preds = model.predict(np.expand_dims(frame, axis=0))[0]
                        Q.append(preds)

                        results = np.array(Q).mean(axis=0)
                        i = (preds > 0.6)[0] #np.argmax(results)

                        label = i

                        text = "Violence: {}".format(label)
                        # print('prediction:', text)
                        # print("print here")
                        # file = open("output/output.txt",'w')
                        # file.write(text)
                        # file.close()

                        color = (0, 255, 0)

                        if label:
                            color = (255, 0, 0) 
                        else:
                            color = (0, 255, 0)

                        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 3)


                        # saving mp4 with labels but cv2.imshow is not working with this notebook
                        if writer is None:
                                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                                writer = cv2.VideoWriter("output/output.mp4", fourcc, 60,
                                        (W, H), True)
                                print("saving video")

                        # print("print here 2")
                        # writer.write(output)
                        #cv2.imshow("Output", output)

                        # fig.add_subplot(8, 3, count)
                        # plt.imshow(output)
                        # plt.show()
                        RESULT.append(label)


                    if limit and count > limit:
                        break

                except:
                    break 
        
        # plt.show()
        print("Cleaning up...")
        if writer is not None:
            writer.release()
        vs.release()

        return "Violence contents exist in the video" if True in RESULT else "There are no violence contents"


def predict_video(video_url):
    try:
      # download the YouTube video
      yt = YouTube(video_url)
      stream = yt.streams.get_highest_resolution()
      video_file = stream.download(output_path="assets/video")
      predict = print_results(video_file,limit=30)
      return predict
    except:
      print("Unexpected error occured")