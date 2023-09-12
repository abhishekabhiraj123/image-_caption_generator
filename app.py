from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import AutoProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the captioning model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    return Image.open(image).convert('RGB')

# Define a function to generate captions for an image
def generate_captions(image):
    # Process the image
    inputs = processor(preprocess_image(image), return_tensors="pt")

    # Generate the output
    outputs = model.generate(
        **inputs,
        max_length=32,
        num_beams=5,
        num_return_sequences=5,
        temperature=1.0,
    )

    # Decode the output and return the captions
    captions = []
    for output in outputs:
        caption = processor.decode(output, skip_special_tokens=True)
        captions.append(caption)

    return captions

# Define a route to serve the HTML file
@app.route('/')
def index():
    return render_template('index1.html')

# Define a route to handle image uploads and generate captions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded'})

    # Save the uploaded image to the uploads folder
    image = request.files['image']
    image.save(f"{app.config['UPLOAD_FOLDER']}/{image.filename}")

    # Generate captions for the uploaded image
    captions = generate_captions(image)

    # Return the generated captions as JSON
    return jsonify({'captions': captions})

if __name__ == '__main__':
    app.run(debug=True)
