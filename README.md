=>Visual Product Matcher

The Visual Product Matcher is an image-based product recommendation system that enables users to find visually similar items by analyzing uploaded images. It leverages a deep learning model (CNN) to extract image features and represent them as high-dimensional embeddings. Using the Nearest Neighbors algorithm with cosine similarity, the system efficiently compares these embeddings to identify the most similar images in the dataset.

The Streamlit interface allows users to easily upload an image, adjust search parameters, and view the most visually similar results in a gallery layout with similarity scores. This project showcases the power of computer vision and machine learning in enhancing e-commerce search capabilities, enabling intuitive and accurate product discovery based on image content rather than textual descriptions.

 =>Project Overview

This project uses a Convolutional Neural Network (CNN) as a feature extractor to convert images into numerical feature vectors (embeddings) that capture essential visual characteristics such as color, texture, and shape.
These embeddings are compared using cosine similarity with the Nearest Neighbors algorithm to retrieve the most similar items.

=>Features

Upload or paste an image URL to search for visually similar products

Adjustable parameters (Top K results and similarity threshold)

Displays matching products with similarity scores

Simple and interactive Streamlit interface

=>Technologies Used

Python

Streamlit

TensorFlow / Keras

scikit-learn

NumPy, Pandas, PIL

=> How to Run

pip install -r requirements.txt

streamlit run app.py

