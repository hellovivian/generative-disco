
from flask import Flask, send_file, request, jsonify, g, render_template, url_for, redirect

app = Flask(__name__)

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
    
def run():

    app.run(host='0.0.0.0', threaded=True, debug=True, port = 7861)
run()
