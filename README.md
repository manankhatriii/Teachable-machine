# Teachable-machine

Manan's Teachable Machine
This is a custom Teachable Machine built using Streamlit and TensorFlow. The app allows users to train their own image classification models directly from the browser, using an interface inspired by Google’s Teachable Machine.

Users can upload images for multiple classes, define class names, and train a model. Once trained, the model can be downloaded as an .h5 file for further use in machine learning projects.

Features
Custom Class Creation: Users can define between 2 to 10 classes, providing names and uploading images for each class.
Image Upload: Supports image uploads in .jpg, .jpeg, and .png formats.
Dynamic Image Classification Training: The model is trained using a pre-trained MobileNetV2 feature extractor via TensorFlow Hub.
Training Progress & Results: Users can view the training progress, including validation accuracy.
Model Download: Once trained, the model can be downloaded as an .h5 file for later use in custom applications.
Tech Stack
Frontend: Streamlit
Backend: TensorFlow, TensorFlow Hub
Model Architecture: Pre-trained MobileNetV2 feature extractor with a custom dense layer for classification.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/teachable-machine.git
cd teachable-machine
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Access the app:

Open the browser and go to http://localhost:8501
Usage
Launch the Streamlit app and follow the instructions:

Select the number of classes you want to create.
Provide names for the classes and upload at least 30 images for each class.
Click on "Start Training" to initiate the training process.
Once training is complete, download the trained model by clicking on "Download Model (.h5)".
The model can be used in other machine learning projects as a .h5 file.

Project Structure
bash
Copy code
.
├── app.py              # Main Streamlit app code
├── requirements.txt    # Python dependencies
├── README.md           # Project readme file
Customization
You can customize the following aspects of the app:

Number of Classes: You can modify the range of classes in the st.selectbox to allow more classes if needed.
Pre-trained Model: The current app uses MobileNetV2 for feature extraction, but you can replace this with other TensorFlow Hub models.
Image Augmentation: The app uses minimal augmentation (rotation, zoom, shift). You can further customize augmentation using the ImageDataGenerator.
Future Enhancements
Add support for video-based training.
Allow users to fine-tune the model parameters directly from the interface.
Enable model inference directly from the app, allowing users to test their trained model with new images.

Contributing
If you find a bug or have a feature request, feel free to open an issue or submit a pull request. Contributions are welcome!


