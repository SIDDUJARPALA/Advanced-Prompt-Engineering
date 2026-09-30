from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from prompt_engineering import call_mistral

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    answer = call_mistral(question)
    return jsonify({"answer": answer})


@app.route("/api/generate", methods=["POST"])
def generate_prompt_api():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip() or str(payload.get("task", "")).strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400

    answer = call_mistral(question)
    return jsonify({"answer": answer, "prompt": question})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
