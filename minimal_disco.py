from flask import Flask, send_file, request, jsonify, g, render_template, url_for, redirect
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/audio'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

music = "./static/audio/cardigan.mp3"
video = "cardigan.mp4"

@app.route("/just_audio")
def just_audio():
    global music
    global video
    
    return render_template("just_audio.html", music = music)

@app.route("/register_music_change", methods=["POST"])
def register_music_change():
    global music
    global video
    music = request.json["music_choice"]
    print(music)
    
    return "completed"

@app.route("/upload_music", methods=["POST"])
def upload_music():
    global music
    global video

    # Check if the POST request has the file part
    if 'newMusic' not in request.files:
        return redirect(request.url)
    file = request.files['newMusic']
    # Check if the file is selected and has an allowed extension
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file"    
    
    # Securely save the file in the ./static/audio/ directory
    music_path = f"{os.getcwd()}/static/audio/{file.filename}"
    print(music_path)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], music_path))

    # Update the global music variable
    
    return redirect(url_for('just_audio'))

def run():
    app.run(host='0.0.0.0', threaded=True, debug=True, port = 7860)
run()