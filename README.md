🌟 Waste Classification using CNN + Streamlit

A Deep Learning–based waste classification system that identifies images of waste into 10 categories using a trained Convolutional Neural Network (CNN) and provides an interactive Streamlit web interface.

This project supports sustainable waste management by automating waste segregation and making recycling more efficient.

 Project Goals

Build a CNN to classify types of waste

Improve accuracy using data augmentation and deeper models

Deploy the trained model using Streamlit

Provide an easy-to-use UI for real-time testing

Support sustainability through AI-based waste segregation

🗂️ Waste Categories

The system classifies waste into 10 classes:

battery

biological

cardboard

clothes

glass

metal

paper

plastic

shoes

trash

🗓️ Week-Wise Progress
🗓️ Week 1 — Data Exploration & Preprocessing
✅ Tasks Completed

Loaded the local dataset garbage-dataset

Verified folder structure and class names

Counted number of images in each class

Displayed sample images

Resized all images to 64×64 for memory efficiency

Normalized pixel values

Created a train-validation split

Documented all work in week1_data_exploration.ipynb

🗓️ Week 2 — Base CNN Model Building & Training
✅ Tasks Completed

Built a simple CNN using TensorFlow/Keras

Trained model on all 10 waste classes

Achieved ~58% validation accuracy

Plotted accuracy & loss graphs

Saved the trained model

Updated notebook: week2_cnn_training.ipynb

🔧 Improvements (Week 2)

Added dropout layer to reduce overfitting

Cleaned and optimized data pipeline

Improved visualization (accuracy/loss curves)

Enhanced model structure and comments

🗓️ Week 3 — Model Improvement + Deployment
🎯 Improvements Applied

Added data augmentation

Built a deeper CNN with BatchNormalization

Trained longer with callbacks

Saved best model as waste_classifier_model.h5

Created a full Streamlit web app

Designed a clean, professional UI

All improvements documented in week3_model_improvement.ipynb

🖥️ Streamlit App Features

Upload image (JPG/PNG)

Predict waste class instantly

Show confidence percentage

Clean and simple UI

Runs locally with Streamlit

🧪 How to Run This Project
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/waste-classification.git
cd waste-classification

2️⃣ Create & Activate Virtual Environment (Recommended)
Windows
python -m venv venv
.\venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Required Libraries
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run streamlit_app/app.py


The app will open at:

👉 http://localhost:8501

📁 Project Folder Structure
waste-classification/
│
├── model/
│   └── waste_classifier_model.h5
│
├── streamlit_app/
│   └── app.py
│
├── notebooks/
│   ├── week1_data_exploration.ipynb
│   ├── week2_cnn_training.ipynb
│   └── week3_model_improvement.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore


⚠️ Note:
The garbage-dataset/ folder is not uploaded to GitHub due to size limitations.

🧰 Tools & Technologies Used

Python 3.10+

TensorFlow / Keras

OpenCV (cv2)

NumPy

Matplotlib

Scikit-learn

Streamlit

Jupyter Notebook

📊 Results

✔ Working CNN waste classifier
✔ Streamlit deployment completed
✔ Improved accuracy using augmentation
✔ Clean UI for demonstration

✍️ Author
