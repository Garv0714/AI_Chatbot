# 🤖 AI Chatbot  

A simple **AI-powered chatbot** built using **Python, Natural Language Processing (NLP), and Deep Learning**.  
It can understand user queries from intents and respond in a conversational way.  

---

## 🚀 Features
- 🧠 Custom-trained on `intents.json` dataset  
- 🤖 Deep learning model built with **TensorFlow / Keras**  
- 🔤 Natural Language Processing with **Tokenizer**  
- 🌐 Web-based chatbot using **Flask**  
- 🖼️ Simple and clean frontend for chatting in browser  

---

## 📂 Project Structure
AI_Chatbot/
│
├── app.py # Flask app to run chatbot (web interface)
├── chatbot_run.py # Run trained chatbot directly in terminal
├── train_chatbot.py # Script to train the chatbot model
├── intents.json # Dataset with intents & responses
├── templates/ # Frontend HTML templates
├── .gitignore # Ignore large/unnecessary files
├── README.md # Project documentation


---

## ⚙️ Installation & Setup  

### 1. Clone the repository
```bash
git clone https://github.com/Garv0714/AI_Chatbot.git
cd AI_Chatbot
2. Create & activate virtual environment (recommended)
bash
Copy code
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Linux/Mac
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Train the chatbot model
bash
Copy code
python train_chatbot.py
5. Run chatbot in terminal
bash
Copy code
python chatbot_run.py
6. Run chatbot as a web app
bash
Copy code
python app.py
👉 Open your browser and go to: http://127.0.0.1:5000/

🧪 Example Usage
User: Hello
Bot: Hi there! How can I help you?

User: Who created you?
Bot: I was created by Garv 😊

🛠️ Tech Stack
Python 3

Flask – Web framework

TensorFlow / Keras – Deep learning model

NLTK / Tokenizer – NLP preprocessing

HTML / CSS – Frontend chat interface

📌 Future Improvements
Add support for context-aware conversations

Connect with external APIs for dynamic responses

Deploy on Heroku / Render / AWS

## 👨‍💻 Author
**Garv Sharma**  

📫 Reach me on:  
- [![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Garv0714)

⭐ If you like this project, don’t forget to star this repo!

