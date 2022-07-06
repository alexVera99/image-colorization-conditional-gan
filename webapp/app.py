import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from ganColorization.gan import Generator
from ganColorization.predict import modelPredictWeb
import torch
import re
import base64

# https://www.youtube.com/watch?v=CSEmUmkfb8Q

# Define a flask app
app = Flask(__name__)

# Load model
model_path = "../Results/cgan_stl_10_100ep_60pts.ckpt"
gan_gen = Generator(img_channels = 4, base_channels = 64, out_channels = 3)
gan_gen.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
gan_gen.eval() # Put in eval model
    
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET'])
def index():
    # Main page
    #return render_template('draw_image.html')
    return render_template('index.html')

# https://stackoverflow.com/questions/41957490/send-canvas-image-data-uint8clampedarray-to-flask-server-via-ajax
@app.route('/hook', methods=['GET', 'POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)
    image_64_decode = base64.b64decode(image_data)
    image_result = open('uploads/mask.jpg', 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)
    return ''

@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        relative_path = os.path.join(
            'uploads', secure_filename(f.filename)
        )
        file_path = os.path.join(
            basepath, relative_path
        )
        f.save(file_path)
        
        # Make prediction
        result = modelPredictWeb(file_path, gan_gen)
        if result["response"]:
            return render_template("show_image.html")
    
    return None

if __name__ == '__main__':
    app.run(debug=True)
