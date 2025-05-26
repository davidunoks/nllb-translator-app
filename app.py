from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load model + tokenizer
model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

model.eval()

@app.route("/", methods=["GET", "POST"])
def translate():
    translation = ""
    if request.method == "POST":
        text = request.form["input_text"]
        lang_code = request.form["lang_code"]

        prompt = f">>{lang_code}<< {text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template("index.html", translation=translation)
