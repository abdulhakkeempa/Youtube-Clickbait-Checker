from flask import Flask, render_template, request
import os, re
from dotenv import load_dotenv

# Strictly related to the web mechanism is written out her all else are called from
# src/ using the OS library [Hope it works for you]
from src.clickbait.thumbnail import Fetch
from src.clickbait.clickbait_dect import Click

#violence content prediction functions
from src.violence.violence_detc import predict_video

load_dotenv()

try:
    YOUTUBE_API_KEY = os.environ.get('API_KEY')
except KeyError:
    print("Did you forget to set API_KEY in .env ? Create a file .env and get the key from GCP")



app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/input')
def input():
  return render_template('input.html')


@app.route('/mature')
def mature():
  return render_template('input-violence.html')

@app.route("/predict",methods=['POST'])
def predict():
  YouTube_Link =  request.form['youtube'].strip()
  # A mistake will render the link unusable

  pattern = re.compile(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)')
  if pattern.match(YouTube_Link) is not None:
    predict = predict_video(YouTube_Link)
    print(predict)
    return render_template('violence-result.html', prediction=predict), 200
  
  else:
    message = """
      The link does not seems to be a Youtube link
      It should be following a definit pattern,
      Use the whole URL and dont use shortend links.
    """
    return render_template('error_5XX.html' , error=message), 500


@app.route('/fetch', methods=['POST'])
def output():
  YouTube_Link =  request.form['youtube'].strip()
  # A mistake will render the link unusable

  pattern = re.compile(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)')
  if pattern.match(YouTube_Link) is not None:
    Fetch(YouTube_Link, YOUTUBE_API_KEY)

    return render_template('download_complete.html', message="Download Complete"), 200
  else:
    message = """
      The link does not seems to be a Youtube link
      It should be following a definit pattern,
      Use the whole URL and dont use shortend links.
    """
    return render_template('error_5XX.html' , error=message), 500
    



@app.route('/clickbait')
def process_clickbait():
  clicks_out = Click()
  print(clicks_out.Frames_similarity_index)
  return render_template('clickbait.html', my_list=clicks_out.Frames_similarity_index)


# Need to reverify if it is a youtube link or not
# To test the same, I'm embedding the youtube link into the web

if __name__ == '__main__':
  app.run()
