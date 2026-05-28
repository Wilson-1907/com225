from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot import ask_question

app = Flask(__name__)

CORS(app)

@app.route("/")
def home():

    return jsonify({
        "message": "COM225 AI Assistant Backend Running"
    })

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        question = data.get("message")

        if not question:

            return jsonify({
                "error": "No message provided"
            }), 400

        answer = ask_question(question)

        return jsonify({
            "response": answer
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
