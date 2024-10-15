from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from werkzeug.utils import secure_filename

# Import your processing functions
from processing.main import main  # Ensure your main function is callable

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'algorithm_outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_videos():
    if 'video1' not in request.files or 'video2' not in request.files:
        return "No file part", 400
    
    video1 = request.files['video1']
    video2 = request.files['video2']
    
    if video1.filename == '' or video2.filename == '':
        return "No selected file", 400
    
    if video1 and allowed_file(video1.filename) and video2 and allowed_file(video2.filename):
        filename1 = secure_filename(video1.filename)
        filename2 = secure_filename(video2.filename)
        video1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        video2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        return redirect(url_for('process_videos', video1=filename1, video2=filename2))
    else:
        return "Invalid file type", 400

@app.route('/process')
def process_videos():
    video1 = request.args.get('video1')
    video2 = request.args.get('video2')
    
    if not video1 or not video2:
        return "Videos not uploaded", 400
    
    video_path1 = os.path.join(app.config['UPLOAD_FOLDER'], video1)
    video_path2 = os.path.join(app.config['UPLOAD_FOLDER'], video2)
    
    # Call your processing function
    try:
        main(video_path1, video_path2, app.config['OUTPUT_FOLDER'])
    except Exception as e:
        return f"An error occurred during processing: {e}", 500
    
    # Assuming your main function saves results in the output directory
    return redirect(url_for('results'))

@app.route('/results')
def results():
    # List output files to display
    output_files = os.listdir(app.config['OUTPUT_FOLDER'])
    return render_template('results.html', files=output_files)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
