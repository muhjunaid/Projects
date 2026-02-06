from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

import os

app = Flask(__name__)

# Load your pre-trained models
models = {
    'resnet50': tf.keras.models.load_model('ResNet50_final.h5'),
    'efficientnet50': tf.keras.models.load_model('EfficientNetB0_final.h5' ),
    'densenet121': tf.keras.models.load_model('ResNet50_final.h5', )
}


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_ai_explanation(result, confidence, model_name):
    """Generate AI explanation using Claude API"""
    
    # OPTION 1: Using Claude API (Uncomment when you have API key)
    """
    try:
        prompt = f'''You are a medical AI assistant. A chest X-ray was analyzed with the following results:
        
        Result: {result}
        Confidence: {confidence}%
        Model: {model_name}
        
        Please provide:
        1. A simple explanation of what this means
        2. What pneumonia is (if detected)
        3. Recommended next steps
        4. Important disclaimers
        
        Keep it concise, empathetic, and easy to understand for patients.'''
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"AI explanation temporarily unavailable: {str(e)}"
    """
    
    # OPTION 2: Fallback template (works without API)
    if "Pneumonia" in result:
        return f"""
        <strong>Understanding Your Results:</strong><br><br>
        
        The AI model has detected signs consistent with pneumonia in your chest X-ray with {confidence:.1f}% confidence.<br><br>
        
        <strong>What is Pneumonia?</strong><br>
        Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.<br><br>
        
        <strong>Recommended Next Steps:</strong><br>
        • Consult with a healthcare professional immediately<br>
        • Share these results with your doctor<br>
        • Follow prescribed treatment if diagnosed<br>
        • Monitor your symptoms closely<br><br>
        
        <strong> Important Disclaimer:</strong><br>
        This AI analysis is a screening tool only and should NOT replace professional medical diagnosis. 
        Always consult qualified healthcare providers for proper diagnosis and treatment.
        """
    else:
        return f"""
        <strong>Understanding Your Results:</strong><br><br>
        
        The AI model did not detect signs of pneumonia in your chest X-ray with {confidence:.1f}% confidence.<br><br>
        
        <strong>What This Means:</strong><br>
        Your X-ray appears normal based on our AI analysis. However, this does not rule out all respiratory conditions.<br><br>
        
        <strong>Recommended Next Steps:</strong><br>
        • If you have symptoms, still consult a healthcare provider<br>
        • Keep monitoring your health<br>
        • Follow up if symptoms worsen<br>
        • Maintain regular health check-ups<br><br>
        
        <strong> Important Disclaimer:</strong><br>
        This AI analysis is a screening tool only. Even with normal results, consult a healthcare 
        professional if you have concerning symptoms.
        """

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        model_name = request.form.get('model', 'resnet50')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Get prediction
        model = models[model_name]
        prediction = model.predict(processed_image)
        confidence = float(prediction[0][0])
        
        # Determine result
        if confidence > 0.5:
            result = 'Pneumonia Detected'
            probability = confidence * 100
        else:
            result = 'Normal'
            probability = (1 - confidence) * 100
        
        # Get AI explanation
        ai_explanation = get_ai_explanation(result, probability, model_name)
        
        return jsonify({
            'result': result,
            'confidence': round(probability, 2),
            'model_used': model_name,
            'ai_explanation': ai_explanation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Additional endpoint for follow-up questions"""
    try:
        data = request.json
        question = data.get('question', '')
        context = data.get('context', {})
        
        # This would use Claude/GPT to answer follow-up questions
        # For now, return a template response
        
        return jsonify({
            'answer': f"Great question! Based on your results, here's what you should know..."
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)