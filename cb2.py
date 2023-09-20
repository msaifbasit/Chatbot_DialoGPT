from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

# create flask app
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the chatbot function using DialoGPT
def chatbot(message):
    input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
    response = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text

# Home route with form to input messages
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['message']
        response_text = chatbot(user_input)
        return render_template('index.html', user_input=user_input, response_text=response_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
