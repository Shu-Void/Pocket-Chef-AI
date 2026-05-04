from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv
from models import db, User, NutritionLog
from functools import wraps
from datetime import date

import os
import uuid
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "pocket-chef-secret-key-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pocket_chef.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# --------------------------------------------------
# INIT DATABASE
# --------------------------------------------------
db.init_app(app)

with app.app_context():
    db.create_all()

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------
MODEL_PATH = "vegetable_detector_yolov8n.pt"
model = YOLO(MODEL_PATH)

# --------------------------------------------------
# LOAD GEMINI CLIENT
# --------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# --------------------------------------------------
# GEMINI CALL WITH AUTO-RETRY
# --------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def gemini_generate(prompt, max_retries=5):
    """Call Gemini with automatic retry on rate limit (429) errors."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return response.text

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = attempt * 20  # 20s, 40s, 60s, 80s, 100s
                logger.warning(
                    f"Gemini rate limited (attempt {attempt}/{max_retries}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise e

    raise Exception(
        "Gemini API rate limit exceeded after multiple retries. "
        "Please wait a minute and try again."
    )

# --------------------------------------------------
# AUTH DECORATOR
# --------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# --------------------------------------------------
# NUTRITION TARGET CALCULATOR (Mifflin-St Jeor)
# --------------------------------------------------
ACTIVITY_MULTIPLIERS = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
}

def calculate_nutrition_targets(user):
    """Calculate personalized daily nutrition targets based on user profile."""

    # Step 1: BMR (Mifflin-St Jeor)
    bmr = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age
    if user.gender == "Female":
        bmr -= 161
    else:
        bmr += 5  # Male or Other

    # Step 2: TDEE
    multiplier = ACTIVITY_MULTIPLIERS.get(user.activity_level, 1.4)
    tdee = bmr * multiplier

    # Step 3: Calorie target based on goal
    goal = user.goal.lower()
    if "lose" in goal or "loose" in goal:
        calorie_target = tdee - 400
    elif "build" in goal or "muscle" in goal:
        calorie_target = tdee + 250
    elif "recompose" in goal or "recomp" in goal:
        calorie_target = tdee  # eat at maintenance
    else:  # maintain
        calorie_target = tdee

    # Step 4: Macro targets
    # Protein: 1.6-2.2g/kg based on goal
    if "build" in goal or "muscle" in goal:
        protein_g = user.weight_kg * 2.0
    elif "lose" in goal or "loose" in goal or "recompose" in goal:
        protein_g = user.weight_kg * 1.8  # high protein to preserve muscle
    else:
        protein_g = user.weight_kg * 1.4

    # Fat: ~25-30% of calories
    fat_g = (calorie_target * 0.27) / 9

    # Carbs: remainder
    protein_cals = protein_g * 4
    fat_cals = fat_g * 9
    carbs_g = (calorie_target - protein_cals - fat_cals) / 4

    # Fiber: 14g per 1000 kcal
    fiber_g = (calorie_target / 1000) * 14

    return {
        "calories": round(calorie_target),
        "protein_g": round(protein_g, 1),
        "carbs_g": round(max(carbs_g, 50), 1),  # minimum 50g carbs
        "fat_g": round(fat_g, 1),
        "fiber_g": round(fiber_g, 1),
    }

# --------------------------------------------------
# DETECT VEGETABLES
# --------------------------------------------------
def detect_vegetables(image_path):
    results = model.predict(
        source=str(image_path),
        conf=0.30,
        save=False,
        verbose=False
    )

    detected = []
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            detected.append(label)

    return list(dict.fromkeys(detected))

# --------------------------------------------------
# BUILD RECIPE PROMPT (with nutrition)
# --------------------------------------------------
def build_recipe_prompt(vegetables, diet=None, cuisine=None, meal_type=None):
    veg_text = ", ".join(vegetables)

    diet_text = f"Strictly follow {diet} diet." if diet else ""
    cuisine_text = f"Focus on {cuisine} cuisine." if cuisine else ""
    meal_text = f"Suitable for {meal_type}." if meal_type else ""

    return f"""
You are a professional chef and nutritionist.

Detected vegetables:
{veg_text}

{diet_text}
{cuisine_text}
{meal_text}

Suggest 3 practical recipes using these vegetables.

Rules:
- Use the detected vegetables prominently.
- Respect diet restrictions strictly.
- Realistic dishes only.
- Estimate nutrition per serving accurately.
- Return ONLY valid JSON.

Format:
{{
  "recipes": [
    {{
      "name": "",
      "description": "",
      "ingredients": [],
      "steps": [],
      "nutrition_per_serving": {{
        "calories": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0,
        "fiber_g": 0
      }}
    }}
  ]
}}
"""

# --------------------------------------------------
# GENERATE RECIPES
# --------------------------------------------------
def generate_recipes(vegetables, diet=None, cuisine=None, meal_type=None):
    prompt = build_recipe_prompt(
        vegetables=vegetables,
        diet=diet,
        cuisine=cuisine,
        meal_type=meal_type
    )

    try:
        text = gemini_generate(prompt)
        data = json.loads(text)
        return data.get("recipes", [])
    except json.JSONDecodeError:
        return []
    except Exception as e:
        raise e  # Let the /analyze route catch it and show error page

# --------------------------------------------------
# GENERATE NUTRITION SUGGESTIONS
# --------------------------------------------------
def generate_nutrition_suggestions(totals, targets, goal):
    prompt = f"""
You are a nutritionist AI assistant.

User's goal: {goal}

Their personalized daily targets:
- Calories: {targets['calories']} kcal
- Protein: {targets['protein_g']} g
- Carbs: {targets['carbs_g']} g
- Fat: {targets['fat_g']} g
- Fiber: {targets['fiber_g']} g

They have consumed so far today:
- Calories: {totals['calories']:.0f} kcal
- Protein: {totals['protein_g']:.1f} g
- Carbs: {totals['carbs_g']:.1f} g
- Fat: {totals['fat_g']:.1f} g
- Fiber: {totals['fiber_g']:.1f} g

Give 3-4 short, practical suggestions for what they should eat next to meet their remaining targets today.
Be specific with food names. Keep each suggestion to 1-2 sentences.
Consider their goal when prioritizing macros.

Return ONLY valid JSON:
{{
  "suggestions": [
    "suggestion text here"
  ]
}}
"""
    try:
        text = gemini_generate(prompt)
        data = json.loads(text)
        return data.get("suggestions", [])
    except Exception:
        return ["Eat a balanced meal with protein, veggies, and whole grains."]

# ==================================================
# ROUTES
# ==================================================

# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# SIGNUP
# --------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("signup.html")

        if len(password) < 4:
            flash("Password must be at least 4 characters.", "error")
            return render_template("signup.html")

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username already taken.", "error")
            return render_template("signup.html")

        # Profile fields
        try:
            height = float(request.form.get("height", 0))
            weight = float(request.form.get("weight", 0))
            age = int(request.form.get("age", 0))
        except (ValueError, TypeError):
            flash("Height, weight, and age must be valid numbers.", "error")
            return render_template("signup.html")

        gender = request.form.get("gender", "").strip()
        activity_level = request.form.get("activity_level", "").strip()
        active_goal = request.form.get("active_goal", "").strip()

        if not all([height, weight, age, gender, activity_level, active_goal]):
            flash("All profile fields are required.", "error")
            return render_template("signup.html")

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            height_cm=height,
            weight_kg=weight,
            age=age,
            gender=gender,
            activity_level=activity_level,
            goal=active_goal
        )
        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        session["username"] = user.username
        flash("Account created! Welcome.", "success")
        return redirect(url_for("home"))

    return render_template("signup.html")

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        session["user_id"] = user.id
        session["username"] = user.username
        flash(f"Welcome back, {user.username}!", "success")
        return redirect(url_for("home"))

    return render_template("login.html")

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("home"))

# --------------------------------------------------
# ANALYZE IMAGE
# --------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_image():
    try:
        image = request.files["image"]

        if image.filename == "":
            return render_template("error.html", message="No image selected.")

        diet = request.form.get("diet")
        cuisine = request.form.get("cuisine")
        meal_type = request.form.get("meal_type")

        # Save image
        file_ext = Path(image.filename).suffix
        unique_name = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_name
        image.save(file_path)

        # Detect vegetables
        vegetables = detect_vegetables(file_path)

        if not vegetables:
            return render_template(
                "error.html",
                message="No vegetables detected in uploaded image."
            )

        # Generate recipes (now with nutrition)
        recipes = generate_recipes(
            vegetables=vegetables,
            diet=diet,
            cuisine=cuisine,
            meal_type=meal_type
        )

        # Store recipes in session for logging
        session["last_recipes"] = recipes
        session["last_meal_type"] = meal_type

        return render_template(
            "result.html",
            image_name=unique_name,
            vegetables=vegetables,
            recipes=recipes
        )

    except Exception as e:
        return render_template("error.html", message=f"Error: {str(e)}")

# --------------------------------------------------
# LOG RECIPE NUTRITION
# --------------------------------------------------
@app.route("/log-recipe", methods=["POST"])
@login_required
def log_recipe():
    try:
        recipe_index = int(request.form.get("recipe_index", 0))
        portions = float(request.form.get("portions", 1))
        recipes = session.get("last_recipes", [])
        meal_type = session.get("last_meal_type", "")

        if recipe_index >= len(recipes):
            flash("Recipe not found.", "error")
            return redirect(url_for("home"))

        recipe = recipes[recipe_index]
        nutrition = recipe.get("nutrition_per_serving", {})

        log = NutritionLog(
            user_id=session["user_id"],
            recipe_name=recipe.get("name", "Unknown"),
            portions=portions,
            calories=nutrition.get("calories", 0) * portions,
            protein_g=nutrition.get("protein_g", 0) * portions,
            carbs_g=nutrition.get("carbs_g", 0) * portions,
            fat_g=nutrition.get("fat_g", 0) * portions,
            fiber_g=nutrition.get("fiber_g", 0) * portions,
            meal_type=meal_type or "Other"
        )

        db.session.add(log)
        db.session.commit()

        flash(f"'{recipe['name']}' ({portions} portion{'s' if portions != 1 else ''}) logged!", "success")
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Could not log recipe: {str(e)}", "error")
        return redirect(url_for("home"))

# --------------------------------------------------
# ADD MANUAL MEAL
# --------------------------------------------------
@app.route("/add-meal", methods=["GET", "POST"])
@login_required
def add_meal():
    if request.method == "POST":
        try:
            recipe_name = request.form.get("recipe_name", "").strip()
            if not recipe_name:
                flash("Dish name is required.", "error")
                return render_template("add_meal.html")

            portions = float(request.form.get("portions", 1))
            calories = float(request.form.get("calories", 0))
            protein_g = float(request.form.get("protein_g", 0))
            carbs_g = float(request.form.get("carbs_g", 0))
            fat_g = float(request.form.get("fat_g", 0))
            fiber_g = float(request.form.get("fiber_g", 0))
            meal_type = request.form.get("meal_type", "Other")

            log = NutritionLog(
                user_id=session["user_id"],
                recipe_name=recipe_name,
                portions=portions,
                calories=calories * portions,
                protein_g=protein_g * portions,
                carbs_g=carbs_g * portions,
                fat_g=fat_g * portions,
                fiber_g=fiber_g * portions,
                meal_type=meal_type
            )

            db.session.add(log)
            db.session.commit()

            flash(f"'{recipe_name}' added to your log!", "success")
            return redirect(url_for("dashboard"))

        except (ValueError, TypeError):
            flash("Please enter valid numbers for nutrition values.", "error")
            return render_template("add_meal.html")

    return render_template("add_meal.html")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    today = date.today()
    user = db.session.get(User, session["user_id"])

    logs = NutritionLog.query.filter_by(
        user_id=session["user_id"],
        logged_date=today
    ).order_by(NutritionLog.logged_at.desc()).all()

    # Calculate totals
    totals = {
        "calories": sum(l.calories for l in logs),
        "protein_g": sum(l.protein_g for l in logs),
        "carbs_g": sum(l.carbs_g for l in logs),
        "fat_g": sum(l.fat_g for l in logs),
        "fiber_g": sum(l.fiber_g for l in logs),
    }

    # Personalized targets
    targets = calculate_nutrition_targets(user)

    # Show suggestions only if redirected from /get-suggestions
    suggestions = session.pop("nutrition_suggestions", [])

    return render_template(
        "dashboard.html",
        logs=logs,
        totals=totals,
        targets=targets,
        user=user,
        suggestions=suggestions,
        today=today
    )

# --------------------------------------------------
# GET AI SUGGESTIONS (on-demand)
# --------------------------------------------------
@app.route("/get-suggestions", methods=["POST"])
@login_required
def get_suggestions():
    today = date.today()
    user = db.session.get(User, session["user_id"])

    logs = NutritionLog.query.filter_by(
        user_id=session["user_id"],
        logged_date=today
    ).all()

    if not logs:
        flash("Log some meals first before getting suggestions.", "warning")
        return redirect(url_for("dashboard"))

    totals = {
        "calories": sum(l.calories for l in logs),
        "protein_g": sum(l.protein_g for l in logs),
        "carbs_g": sum(l.carbs_g for l in logs),
        "fat_g": sum(l.fat_g for l in logs),
        "fiber_g": sum(l.fiber_g for l in logs),
    }

    targets = calculate_nutrition_targets(user)
    suggestions = generate_nutrition_suggestions(totals, targets, user.goal)
    session["nutrition_suggestions"] = suggestions

    return redirect(url_for("dashboard"))

# --------------------------------------------------
# DELETE LOG ENTRY
# --------------------------------------------------
@app.route("/delete-log/<int:log_id>", methods=["POST"])
@login_required
def delete_log(log_id):
    log = db.session.get(NutritionLog, log_id)

    if log and log.user_id == session["user_id"]:
        db.session.delete(log)
        db.session.commit()
        flash("Entry removed.", "success")
    else:
        flash("Entry not found.", "error")

    return redirect(url_for("dashboard"))

# --------------------------------------------------
# SERVE UPLOADED IMAGES
# --------------------------------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)

# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)