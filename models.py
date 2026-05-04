from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    height_cm = db.Column(db.Float, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)  # e.g., 'Male', 'Female', 'Other'
    activity_level = db.Column(db.String(50), nullable=False) # e.g., 'Sedentary', 'Light', etc.
    goal = db.Column(db.String(50), nullable=False)           # e.g., 'Lose Fat', 'Build Muscle'
    
    logs = db.relationship("NutritionLog", backref="user", lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username}>"


class NutritionLog(db.Model):
    __tablename__ = "nutrition_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    recipe_name = db.Column(db.String(200), nullable=False)
    portions = db.Column(db.Float, default=1)
    calories = db.Column(db.Float, default=0)
    protein_g = db.Column(db.Float, default=0)
    carbs_g = db.Column(db.Float, default=0)
    fat_g = db.Column(db.Float, default=0)
    fiber_g = db.Column(db.Float, default=0)

    meal_type = db.Column(db.String(50)) # e.g., 'Breakfast', 'Lunch'
    logged_date = db.Column(db.Date, default=date.today)
    logged_at = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<NutritionLog {self.recipe_name} | {self.logged_date}>"