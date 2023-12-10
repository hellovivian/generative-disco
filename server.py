from flask import Flask, jsonify

app = Flask(__name__)

music = "./static/audio/cardigan.mp3"
video = "cardigan.mp4"


@app.route("/just_audio")
def just_audio():
    global music
    global video

    music_info = {"music": music, "video": video}

    return jsonify(music_info)


def run():
    app.run(host="0.0.0.0", threaded=True, debug=True, port=7870)


run()
