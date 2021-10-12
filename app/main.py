from flask import Flask, request, jsonify

from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
