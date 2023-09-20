import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Create the Gradio interface
iface = gr.Interface(fn=chatbot, inputs="text", outputs="text")

# Start the interface
iface.launch()

