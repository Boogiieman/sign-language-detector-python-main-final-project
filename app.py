from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_inference')
def run_inference():
    subprocess.run(['python', 'inference_classifier.py'])
    return 'Inference Started'

@app.route('/mal_run_inference')
def mal_run_inference():
    subprocess.run(['python', 'regional\Malayalam\mal_inference_classifier.py'])
    return 'Inference Started'

@app.route('/cust_run_inference')
def cust_run_inference():
    subprocess.run(['python', 'custom_mode\cust_inference_classifier.py'])
    return 'Inference Started'

if __name__ == '__main__':
    app.run(debug=True)
