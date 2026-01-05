# 🖐️ SGT: Smart Gesture Translator

**A Real-Time AI System that gives a voice to Sign Language.**

> **Built for Dimension X Hackathon 2026**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%201.5%20Pro-4285F4)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-green)

## 💡 Overview
**SGT (Smart Gesture Translator)** is an assistive technology project designed to bridge the communication gap between the mute community and the general public.

Unlike standard sign language detectors that just output letters, **SGT uses Generative AI (Google Gemini 1.5 Pro)** to intelligently reconstruct broken sentences and **Google TTS** to speak them aloud in a natural human voice.

## 🚀 Key Features
* **Real-Time Detection:** Uses **MediaPipe** and a custom **Deep Neural Network (DNN)** to detect hand signs instantly.
* **Generative AI Correction:** Integrated **Google Gemini 1.5 Pro** to fix grammar, split words (e.g., "IAMFINE" -> "I am fine"), and correct spelling.
* **Natural Voice Output:** Uses **Google Text-to-Speech (gTTS)** for high-quality audio feedback.
* **Smart "Safety Net":** Zero-latency detection for common phrases like "Hello World" or "Emergency."

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **AI Model:** TensorFlow/Keras (Custom trained CNN/DNN)
* **LLM Integration:** Google Generative AI (Gemini 1.5 Pro)
* **Audio:** gTTS (Google Text-to-Speech)

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ironman1947/SGT-Smart-Gesture-Translator.git](https://github.com/YOUR_USERNAME/SGT-Smart-Gesture-Translator.git)
    cd SGT-Smart-Gesture-Translator
    ```

2.  **Install Dependencies**
    ```bash
    pip install opencv-python mediapipe numpy tensorflow google-generativeai gTTS playsound python-dotenv
    ```

3.  **Setup API Key (Crucial Step)**
    * Create a file named `.env` in the project root.
    * Add your Google Gemini API key inside it:
        ```ini
        GEMINI_API_KEY=AIzaSy... (Paste your key here)
        ```

4.  **Run the System**
    ```bash
    python Real.py
    ```

## 🎮 Controls
* **Sign A-Z:** Spell your words.
* **Press 'C':** Confirm a letter.
* **Press 'T':** Add a space (Optional, AI handles this too!).
* **Press 'S':** **SPEAK** (Triggers Gemini AI + Google Voice).
* **ESC:** Exit.

## 🤖 AI Model Details
The system relies on a custom-trained Keras model (`SGT.h5`) trained on a dataset of 26 alphabets + special command gestures.

## 👥 The Team
* **Om Chougule** 
* **Harsh Desai** - *Team Lead*
* **Rohan Ilke**
* **Nikhil Gosai**

