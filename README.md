# 🍽️ Pocket Chef AI

An AI-powered nutrition and meal tracking web application that helps users analyze food images, generate recipes, and track daily nutrition intelligently.

---

## 🚀 Features

* 🧠 **AI Recipe Generation**

  * Upload an image
  * Detect vegetables using YOLOv8
  * Generate recipes using Google Gemini AI

* 📊 **Nutrition Tracking Dashboard**

  * Track calories, protein, carbs, fat, fiber
  * View daily totals

* 🎯 **Personalized Nutrition Targets**

  * Based on height, weight, age, activity level, and goal

* 🤖 **AI Nutrition Suggestions**

  * Get smart suggestions to meet daily targets

* 📝 **Manual Meal Logging**

  * Add custom meals with nutrition values

* 🔐 **Authentication System**

  * Secure signup/login with hashed passwords

---

## 🧩 Tech Stack

* **Backend:** Flask
* **Database:** SQLite + SQLAlchemy
* **AI Models:**

  * YOLOv8 (Vegetable Detection)
  * Google Gemini API (Recipe + Suggestions)
* **Frontend:** HTML (Jinja Templates)

---

## 📁 Project Structure

```bash
Pocket Chef AI/
│
├── app.py
├── models.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   ├── signup.html
│   ├── login.html
│   ├── dashboard.html
│   ├── result.html
│   └── add_meal.html
│
├── uploads/              # User uploaded images
├── instance/             # SQLite database
├── Other files/          # Training and testing files for YOLO model
│
├── .env                  # Environment variables (not pushed)
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/pocket-chef-ai.git
cd pocket-chef-ai
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

---

### 5. Run the app

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. Upload food image
2. YOLO detects vegetables
3. Gemini AI generates recipes + nutrition
4. User logs meals
5. Dashboard calculates totals vs targets
6. AI suggests next meals

---

## ⚠️ Important Notes

* Do NOT upload:

  * `venv/`
  * `.env`
  * large dataset files
* Model file (`.pt`) can be replaced with your own trained model

---

## 🔮 Future Improvements

* Mobile responsive UI
* Barcode scanning for packaged foods
* Meal planning automation
* Integration with fitness trackers
* Better UI/UX

---

## 👨‍💻 Author

**Sudhanshu Prabhat**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
