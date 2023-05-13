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
          print("Unable to load the model,please restart the server")

        vs = cv2.VideoCapture(video)
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
                        frame = cv2.resize(frame, (128, 128)).astype("float16")
                        frame = frame.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                        preds = model([frame])

                        i = (preds > 0.6)[0] #np.argmax(results)

                        label = i
                        RESULT.append(label)


                    if limit and count > limit:
                        break

                except:
                    break 
        
        # plt.show()
        print("Cleaning up...")
        # if writer is not None:
        #     writer.release()
        vs.release()

        return "Violence contents exist in the video" if True in RESULT else "There are no violence contents"


def predict_video(video_url):
    try:
      # download the YouTube video
      yt = YouTube(video_url)
      stream = yt.streams.get_highest_resolution()
      print(stream)
      video_file = stream.download(output_path="assets/video")
      print(video_file)
      predict = print_results(video_file)
      return predict
    except Exception as err:
      print(err)
      print("Unexpected error occured")