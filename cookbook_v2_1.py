"""
🍳 Smart Cookbook v2 — What Can I Cook?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enter the ingredients you have at home and discover
what you can cook — with quantities, nutrition info,
allergen warnings, serving scaler and more.

APIs (all free):
━━━━━━━━━━━━━━━━
1. TheMealDB   → recipes, ingredients + quantities, instructions, photos
   https://www.themealdb.com/api/json/v1/1/  (no key needed)


Allergen detection:
━━━━━━━━━━━━━━━━━━
Rule-based detection of all 14 EU mandatory allergens
from ingredient names — no API needed, works offline.

ML component:
━━━━━━━━━━━━
- KNN similarity: find recipes similar to your favourites
- Coverage-based ranking: sort by % of ingredients you have
- "1 ingredient away" detection

Hardcoded:
━━━━━━━━━━
- 15 Swiss/European bonus recipes
- Allergen keyword mappings (EU 14 allergens)
- Difficulty ratings per recipe category

AI Assistance: Developed with Claude (Anthropic), April 2026 | claude.ai
"""

import streamlit as st
import requests
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🍳 Smart Cookbook",
    page_icon="🍳",
    layout="wide"
)

st.markdown("""
<style>
    .ingredient-tag {
        display: inline-block;
        background: #e8f5e9;
        border: 1px solid #81c784;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.85rem;
    }
    .missing-tag {
        display: inline-block;
        background: #fce4ec;
        border: 1px solid #e57373;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.85rem;
    }
    .allergen-tag {
        display: inline-block;
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EU 14 MANDATORY ALLERGENS
# Keyword mapping for rule-based detection.
# No API needed — scan ingredient names.
# Source: EU Food Information Regulation 1169/2011
# ─────────────────────────────────────────────

ALLERGEN_KEYWORDS = {
    "🌾 Gluten":      ["wheat", "flour", "bread", "pasta", "barley",
                       "rye", "oats", "couscous", "semolina", "spelt",
                       "soy sauce", "beer", "breadcrumbs"],
    "🥛 Dairy":       ["milk", "cheese", "butter", "cream", "yogurt",
                       "parmesan", "mozzarella", "cheddar", "gruyere",
                       "emmental", "brie", "feta", "ricotta", "ghee",
                       "whey", "lactose", "sour cream", "creme fraiche"],
    "🥚 Eggs":        ["egg", "eggs", "mayonnaise", "meringue",
                       "hollandaise", "aioli"],
    "🐟 Fish":        ["fish", "salmon", "tuna", "cod", "anchovy",
                       "sardine", "mackerel", "trout", "bass", "tilapia",
                       "worcestershire sauce"],
    "🦐 Shellfish":   ["shrimp", "prawn", "crab", "lobster", "crayfish",
                       "scallop", "oyster", "clam", "mussel"],
    "🥜 Peanuts":     ["peanut", "groundnut", "peanut butter",
                       "peanut oil", "satay"],
    "🌰 Tree Nuts":   ["almond", "walnut", "cashew", "pistachio",
                       "hazelnut", "pecan", "macadamia", "brazil nut",
                       "chestnut", "pine nut", "marzipan", "praline"],
    "🫘 Soy":         ["soy", "soya", "tofu", "tempeh", "edamame",
                       "miso", "soy sauce", "tamari"],
    "🌱 Celery":      ["celery", "celeriac", "celery salt", "celery seed"],
    "🌻 Sesame":      ["sesame", "tahini", "sesame oil", "sesame seed",
                       "hummus"],
    "🟡 Mustard":     ["mustard", "mustard seed", "mustard oil",
                       "mustard powder"],
    "🫐 Sulphites":   ["wine", "vinegar", "dried fruit", "beer",
                       "cider", "grape juice"],
    "🦑 Molluscs":    ["squid", "octopus", "snail", "abalone", "clam",
                       "mussel", "oyster", "scallop"],
    "🌿 Lupin":       ["lupin", "lupin flour", "lupin seed"],
}

# ─────────────────────────────────────────────
# HARDCODED SWISS / EUROPEAN BONUS RECIPES
# ─────────────────────────────────────────────

BONUS_RECIPES = [
    {
        "id": "swiss_001", "name": "Rösti",
        "category": "Side", "area": "Swiss",
        "difficulty": "Easy", "time_mins": 30,
        "ingredients": ["potatoes", "butter", "salt", "pepper", "onion"],
        "measures":    ["500g", "2 tbsp", "1 tsp", "to taste", "1 medium"],
        "instructions": (
            "1. Grate potatoes coarsely and squeeze out excess water.\n"
            "2. Season with salt and pepper, mix in grated onion.\n"
            "3. Heat butter in a non-stick pan over medium heat.\n"
            "4. Press potato mixture into pan to form a flat cake.\n"
            "5. Cook 10-12 min per side until golden and crispy.\n"
            "6. Slide onto plate and serve immediately."
        ),
        "image": "", "tags": "Swiss,Potato,Vegetarian",
    },
    {
        "id": "swiss_002", "name": "Cheese Fondue",
        "category": "Dinner", "area": "Swiss",
        "difficulty": "Medium", "time_mins": 25,
        "ingredients": ["gruyere cheese", "emmental cheese", "white wine",
                        "garlic", "bread", "cornstarch", "kirsch", "nutmeg"],
        "measures":    ["300g", "200g", "300ml", "1 clove", "1 loaf",
                        "1 tbsp", "2 tbsp", "pinch"],
        "instructions": (
            "1. Rub fondue pot with halved garlic clove.\n"
            "2. Pour wine into pot and heat gently.\n"
            "3. Gradually add grated cheese, stirring in figure-8 motion.\n"
            "4. Mix cornstarch with kirsch, add to cheese.\n"
            "5. Season with nutmeg and pepper.\n"
            "6. Keep warm over low heat. Dip bread cubes to serve."
        ),
        "image": "", "tags": "Swiss,Cheese,Dinner Party",
    },
    {
        "id": "swiss_003", "name": "Zürcher Geschnetzeltes",
        "category": "Dinner", "area": "Swiss",
        "difficulty": "Medium", "time_mins": 35,
        "ingredients": ["veal", "mushrooms", "cream", "white wine",
                        "onion", "butter", "lemon", "parsley"],
        "measures":    ["400g", "200g", "150ml", "100ml",
                        "1 medium", "2 tbsp", "1/2", "handful"],
        "instructions": (
            "1. Cut veal into thin strips, season.\n"
            "2. Sauté onions in butter until soft.\n"
            "3. Add veal, cook quickly 2-3 min over high heat.\n"
            "4. Add mushrooms, cook 3 more minutes.\n"
            "5. Deglaze with wine, reduce by half.\n"
            "6. Add cream, simmer 5 minutes.\n"
            "7. Finish with lemon juice and parsley.\n"
            "8. Serve with Rösti or noodles."
        ),
        "image": "", "tags": "Swiss,Veal,Zurich",
    },
    {
        "id": "swiss_004", "name": "Birchermüesli",
        "category": "Breakfast", "area": "Swiss",
        "difficulty": "Easy", "time_mins": 10,
        "ingredients": ["oats", "milk", "apple", "lemon juice",
                        "honey", "yogurt", "nuts", "berries"],
        "measures":    ["100g", "150ml", "1 large", "1 tsp",
                        "1 tbsp", "100g", "30g", "handful"],
        "instructions": (
            "1. Soak oats in milk overnight in the fridge.\n"
            "2. Grate apple, mix with lemon juice.\n"
            "3. Combine oats, apple, yogurt, honey.\n"
            "4. Top with berries and nuts.\n"
            "5. Serve cold."
        ),
        "image": "", "tags": "Swiss,Breakfast,Healthy,Vegetarian",
    },
    {
        "id": "swiss_005", "name": "Raclette",
        "category": "Dinner", "area": "Swiss",
        "difficulty": "Easy", "time_mins": 20,
        "ingredients": ["raclette cheese", "potatoes", "pickles",
                        "pearl onions", "pepper"],
        "measures":    ["200g per person", "3-4 medium", "as needed",
                        "as needed", "to taste"],
        "instructions": (
            "1. Boil potatoes until tender.\n"
            "2. Melt raclette cheese under grill or machine.\n"
            "3. Scrape cheese onto potatoes.\n"
            "4. Serve with pickles and pearl onions.\n"
            "5. Season with pepper."
        ),
        "image": "", "tags": "Swiss,Cheese,Winter",
    },
    {
        "id": "eur_001", "name": "Pasta Carbonara",
        "category": "Dinner", "area": "Italian",
        "difficulty": "Medium", "time_mins": 20,
        "ingredients": ["spaghetti", "eggs", "pancetta",
                        "parmesan", "black pepper", "salt"],
        "measures":    ["200g", "3 large", "150g", "80g", "1 tsp", "to taste"],
        "instructions": (
            "1. Cook spaghetti in salted water until al dente.\n"
            "2. Fry pancetta until crispy.\n"
            "3. Mix eggs with parmesan and black pepper.\n"
            "4. Remove pan from heat, add drained pasta.\n"
            "5. Pour egg mixture over, toss quickly.\n"
            "6. Add pasta water to loosen. Serve immediately."
        ),
        "image": "", "tags": "Italian,Pasta,Quick",
    },
    {
        "id": "eur_002", "name": "French Onion Soup",
        "category": "Starter", "area": "French",
        "difficulty": "Medium", "time_mins": 60,
        "ingredients": ["onion", "beef broth", "butter", "white wine",
                        "bread", "gruyere cheese", "thyme", "bay leaf"],
        "measures":    ["4 large", "1 litre", "3 tbsp", "150ml",
                        "4 slices", "100g", "2 sprigs", "2 leaves"],
        "instructions": (
            "1. Caramelise sliced onions in butter 40 min.\n"
            "2. Add wine, thyme, bay leaf. Cook 5 min.\n"
            "3. Add broth, simmer 15 min. Season.\n"
            "4. Ladle into oven-safe bowls.\n"
            "5. Top with toasted bread and gruyère.\n"
            "6. Grill until cheese is bubbly."
        ),
        "image": "", "tags": "French,Soup,Winter",
    },
    {
        "id": "eur_003", "name": "Spanish Omelette",
        "category": "Breakfast", "area": "Spanish",
        "difficulty": "Easy", "time_mins": 30,
        "ingredients": ["eggs", "potatoes", "onion", "olive oil", "salt"],
        "measures":    ["6 large", "3 medium", "1 large", "100ml", "1 tsp"],
        "instructions": (
            "1. Slice potatoes and onions thinly.\n"
            "2. Fry slowly in olive oil until soft.\n"
            "3. Beat eggs with salt.\n"
            "4. Drain potatoes, add to eggs.\n"
            "5. Cook until bottom set, flip with plate.\n"
            "6. Cook other side 3 minutes."
        ),
        "image": "", "tags": "Spanish,Eggs,Vegetarian",
    },
    {
        "id": "eur_004", "name": "Beef Stroganoff",
        "category": "Dinner", "area": "Russian",
        "difficulty": "Medium", "time_mins": 40,
        "ingredients": ["beef", "mushrooms", "sour cream", "onion",
                        "butter", "beef broth", "mustard", "paprika"],
        "measures":    ["400g", "200g", "200ml", "1 large",
                        "2 tbsp", "150ml", "1 tsp", "1 tsp"],
        "instructions": (
            "1. Cut beef into strips, season.\n"
            "2. Sauté onions until golden. Set aside.\n"
            "3. Brown beef in batches. Set aside.\n"
            "4. Cook mushrooms in same pan.\n"
            "5. Return beef and onions. Add broth, mustard, paprika.\n"
            "6. Simmer 5 min. Stir in sour cream.\n"
            "7. Serve over noodles or rice."
        ),
        "image": "", "tags": "Russian,Beef,Comfort Food",
    },
    {
        "id": "eur_005", "name": "Minestrone",
        "category": "Starter", "area": "Italian",
        "difficulty": "Easy", "time_mins": 45,
        "ingredients": ["tomatoes", "carrot", "celery", "onion", "garlic",
                        "zucchini", "pasta", "vegetable broth",
                        "olive oil", "parmesan", "basil"],
        "measures":    ["400g tin", "2 medium", "2 sticks", "1 large",
                        "3 cloves", "1 medium", "80g", "1.5 litres",
                        "3 tbsp", "50g", "handful"],
        "instructions": (
            "1. Sauté onion, carrot, celery in olive oil 5 min.\n"
            "2. Add garlic and tomatoes. Cook 3 min.\n"
            "3. Pour in broth, bring to boil.\n"
            "4. Add zucchini and pasta. Simmer 12 min.\n"
            "5. Season, add basil.\n"
            "6. Serve with parmesan."
        ),
        "image": "", "tags": "Italian,Soup,Vegetarian",
    },
    {
        "id": "eur_006", "name": "Shakshuka",
        "category": "Breakfast", "area": "Middle Eastern",
        "difficulty": "Easy", "time_mins": 25,
        "ingredients": ["eggs", "tomatoes", "bell pepper", "onion",
                        "garlic", "cumin", "paprika", "olive oil",
                        "feta cheese"],
        "measures":    ["4 large", "400g tin", "1 large", "1 medium",
                        "3 cloves", "1 tsp", "1 tsp", "2 tbsp", "80g"],
        "instructions": (
            "1. Sauté onion and pepper in olive oil.\n"
            "2. Add garlic, cumin, paprika. Cook 1 min.\n"
            "3. Add tomatoes, simmer 10 min.\n"
            "4. Make wells, crack eggs in.\n"
            "5. Cover, cook 5-8 min until whites set.\n"
            "6. Top with feta. Serve with bread."
        ),
        "image": "", "tags": "Middle Eastern,Eggs,Vegetarian",
    },
    {
        "id": "eur_007", "name": "Lentil Soup",
        "category": "Starter", "area": "Mediterranean",
        "difficulty": "Easy", "time_mins": 40,
        "ingredients": ["lentils", "onion", "carrot", "garlic", "cumin",
                        "turmeric", "olive oil", "lemon", "vegetable broth"],
        "measures":    ["250g", "1 large", "2 medium", "3 cloves", "1 tsp",
                        "1/2 tsp", "3 tbsp", "1 whole", "1.2 litres"],
        "instructions": (
            "1. Sauté onion and carrot in olive oil.\n"
            "2. Add garlic, cumin, turmeric. Cook 1 min.\n"
            "3. Add rinsed lentils and broth. Boil.\n"
            "4. Simmer 25 min until soft.\n"
            "5. Blend partially for creamy texture.\n"
            "6. Finish with lemon juice and olive oil."
        ),
        "image": "", "tags": "Mediterranean,Vegan,Healthy",
    },
    {
        "id": "eur_008", "name": "Banana Pancakes",
        "category": "Breakfast", "area": "American",
        "difficulty": "Easy", "time_mins": 15,
        "ingredients": ["banana", "eggs", "flour", "milk",
                        "baking powder", "butter", "honey"],
        "measures":    ["2 ripe", "2 large", "120g", "150ml",
                        "1 tsp", "1 tbsp", "2 tbsp"],
        "instructions": (
            "1. Mash banana in a bowl.\n"
            "2. Beat in eggs, add milk.\n"
            "3. Stir in flour and baking powder.\n"
            "4. Heat butter in pan.\n"
            "5. Pour small ladles, cook 2 min per side.\n"
            "6. Serve with honey."
        ),
        "image": "", "tags": "Breakfast,Sweet,Easy",
    },
    {
        "id": "eur_009", "name": "Garlic Butter Shrimp Pasta",
        "category": "Dinner", "area": "Italian",
        "difficulty": "Easy", "time_mins": 20,
        "ingredients": ["shrimp", "pasta", "garlic", "butter",
                        "white wine", "parsley", "lemon", "chili flakes"],
        "measures":    ["300g", "200g", "4 cloves", "3 tbsp",
                        "100ml", "handful", "1/2", "pinch"],
        "instructions": (
            "1. Cook pasta until al dente.\n"
            "2. Melt butter, sauté garlic 1 min.\n"
            "3. Add shrimp, cook 2 min per side.\n"
            "4. Deglaze with wine, cook 2 min.\n"
            "5. Add pasta, lemon juice and parsley.\n"
            "6. Finish with chili flakes."
        ),
        "image": "", "tags": "Italian,Seafood,Quick",
    },
    {
        "id": "eur_010", "name": "Vegetable Stir Fry",
        "category": "Dinner", "area": "Asian",
        "difficulty": "Easy", "time_mins": 15,
        "ingredients": ["broccoli", "carrot", "bell pepper", "garlic",
                        "soy sauce", "sesame oil", "ginger",
                        "rice", "cornstarch"],
        "measures":    ["200g", "2 medium", "1 large", "3 cloves",
                        "3 tbsp", "1 tbsp", "1 tsp fresh", "200g", "1 tsp"],
        "instructions": (
            "1. Cook rice according to packet.\n"
            "2. Heat sesame oil in wok over high heat.\n"
            "3. Add garlic and ginger, stir 30 sec.\n"
            "4. Add vegetables, stir fry 4-5 min.\n"
            "5. Mix soy sauce with cornstarch, pour over.\n"
            "6. Toss until coated. Serve over rice."
        ),
        "image": "", "tags": "Asian,Vegan,Quick,Healthy",
    },
]


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect("cookbook_v2.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_recipes (
            meal_id      TEXT PRIMARY KEY,
            name         TEXT,
            category     TEXT,
            area         TEXT,
            instructions TEXT,
            image_url    TEXT,
            tags         TEXT,
            ingredients  TEXT,
            measures     TEXT,
            youtube_url  TEXT,
            fetched_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id    TEXT,
            recipe_name  TEXT,
            source       TEXT,
            saved_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
            ingredients  TEXT,
            diet_filter  TEXT,
            results_count INTEGER,
            top_result   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            recipe_id    TEXT PRIMARY KEY,
            recipe_name  TEXT,
            rating       INTEGER,
            notes        TEXT,
            rated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_api_recipe(meal):
    """Parse TheMealDB response and save to DB including measures."""
    ingredients, measures = [], []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        msr = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            ingredients.append(ing.lower())
            measures.append(msr)
    conn = sqlite3.connect("cookbook_v2.db")
    conn.execute("""
        INSERT OR REPLACE INTO api_recipes VALUES
        (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
    """, (
        meal["idMeal"], meal["strMeal"],
        meal.get("strCategory",""), meal.get("strArea",""),
        meal.get("strInstructions",""), meal.get("strMealThumb",""),
        meal.get("strTags",""),
        "|".join(ingredients), "|".join(measures),
        meal.get("strYoutube",""),
    ))
    conn.commit()
    conn.close()






def load_cached_recipes():
    conn = sqlite3.connect("cookbook_v2.db")
    df = pd.read_sql_query("SELECT * FROM api_recipes", conn)
    conn.close()
    return df


def save_favourite(recipe_id, recipe_name, source):
    conn = sqlite3.connect("cookbook_v2.db")
    conn.execute(
        "INSERT INTO favourites (recipe_id,recipe_name,source) VALUES (?,?,?)",
        (recipe_id, recipe_name, source))
    conn.commit()
    conn.close()


def load_favourites():
    conn = sqlite3.connect("cookbook_v2.db")
    df = pd.read_sql_query(
        "SELECT * FROM favourites ORDER BY saved_at DESC", conn)
    conn.close()
    return df


def save_rating(recipe_id, recipe_name, rating, notes):
    conn = sqlite3.connect("cookbook_v2.db")
    conn.execute(
        "INSERT OR REPLACE INTO user_ratings VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
        (recipe_id, recipe_name, rating, notes))
    conn.commit()
    conn.close()


def load_ratings():
    conn = sqlite3.connect("cookbook_v2.db")
    df = pd.read_sql_query(
        "SELECT * FROM user_ratings ORDER BY rated_at DESC", conn)
    conn.close()
    return df


def log_search(ingredients, diet, count, top):
    conn = sqlite3.connect("cookbook_v2.db")
    conn.execute(
        "INSERT INTO search_log (ingredients,diet_filter,results_count,top_result)"
        " VALUES (?,?,?,?)",
        (", ".join(ingredients), diet, count, top))
    conn.commit()
    conn.close()


def load_search_log():
    conn = sqlite3.connect("cookbook_v2.db")
    df = pd.read_sql_query(
        "SELECT * FROM search_log ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()
    return df


# ─────────────────────────────────────────────
# THEMEALDB API
# ─────────────────────────────────────────────

BASE = "https://www.themealdb.com/api/json/v1/1"


@st.cache_data(ttl=86400)
def api_filter_by_ingredient(ingredient):
    try:
        r = requests.get(f"{BASE}/filter.php",
                         params={"i": ingredient}, timeout=10)
        return r.json().get("meals") or []
    except Exception:
        return []


@st.cache_data(ttl=86400)
def api_lookup_by_id(meal_id):
    try:
        r = requests.get(f"{BASE}/lookup.php",
                         params={"i": meal_id}, timeout=10)
        meals = r.json().get("meals")
        return meals[0] if meals else None
    except Exception:
        return None


@st.cache_data(ttl=86400)
def api_search_by_name(query):
    try:
        r = requests.get(f"{BASE}/search.php",
                         params={"s": query}, timeout=10)
        return r.json().get("meals") or []
    except Exception:
        return []


@st.cache_data(ttl=86400)
def api_get_random():
    try:
        r = requests.get(f"{BASE}/random.php", timeout=10)
        meals = r.json().get("meals")
        return meals[0] if meals else None
    except Exception:
        return None


@st.cache_data(ttl=86400)
def api_get_categories():
    try:
        r = requests.get(f"{BASE}/categories.php", timeout=10)
        return r.json().get("categories") or []
    except Exception:
        return []


# ─────────────────────────────────────────────
# ALLERGEN DETECTION
# Rule-based scan of ingredient names against
# EU 14 mandatory allergen keyword list.
# No API needed — works offline.
# ─────────────────────────────────────────────

def detect_allergens(ingredients):
    """
    Detect allergens by scanning ingredient names
    against the EU 14 mandatory allergen keyword list.
    Returns list of detected allergen names.
    """
    detected = []
    ing_text = " ".join(i.lower() for i in ingredients)
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        for kw in keywords:
            if kw in ing_text:
                detected.append(allergen)
                break
    return detected


# ─────────────────────────────────────────────
# RECIPE FETCHING AND MATCHING
# ─────────────────────────────────────────────

def fetch_recipes_for_ingredients(user_ingredients):
    """
    For each ingredient, fetch matching meals from TheMealDB.
    Look up full details. Cache in SQLite.
    Returns list of recipe dicts with ingredients + measures.
    """
    seen_ids = set()
    full_meals = []
    cached_df = load_cached_recipes()
    cached_ids = set(cached_df["meal_id"].tolist()) \
        if not cached_df.empty else set()

    for ingredient in user_ingredients[:6]:
        matches = api_filter_by_ingredient(ingredient.strip())
        for m in matches[:5]:
            meal_id = m["idMeal"]
            if meal_id in seen_ids:
                continue
            seen_ids.add(meal_id)

            if meal_id in cached_ids:
                row = cached_df[cached_df["meal_id"] == meal_id].iloc[0]
                ings = row["ingredients"].split("|") \
                    if row["ingredients"] else []
                msrs = row["measures"].split("|") \
                    if pd.notna(row["measures"]) and row["measures"] else []
                full_meals.append({
                    "source": "api", "id": meal_id,
                    "name":   row["name"],
                    "category": str(row["category"] or ""),
                    "area":   str(row["area"] or ""),
                    "instructions": row["instructions"],
                    "image":  row["image_url"],
                    "tags":   str(row["tags"] or ""),
                    "ingredients": ings,
                    "measures":    msrs,
                    "youtube": row["youtube_url"],
                    "difficulty": "Medium", "time_mins": None,
                })
            else:
                meal = api_lookup_by_id(meal_id)
                if meal:
                    save_api_recipe(meal)
                    ings, msrs = [], []
                    for i in range(1, 21):
                        ing = (meal.get(f"strIngredient{i}") or "").strip()
                        msr = (meal.get(f"strMeasure{i}") or "").strip()
                        if ing:
                            ings.append(ing.lower())
                            msrs.append(msr)
                    full_meals.append({
                        "source": "api", "id": meal_id,
                        "name":   meal["strMeal"],
                        "category": str(meal.get("strCategory") or ""),
                        "area":   str(meal.get("strArea") or ""),
                        "instructions": meal.get("strInstructions",""),
                        "image":  meal.get("strMealThumb",""),
                        "tags":   str(meal.get("strTags") or ""),
                        "ingredients": ings,
                        "measures":    msrs,
                        "youtube": meal.get("strYoutube",""),
                        "difficulty": "Medium", "time_mins": None,
                    })
    return full_meals


def get_bonus_recipes():
    return [{
        "source": "hardcoded", "id": r["id"], "name": r["name"],
        "category": r["category"], "area": r["area"],
        "instructions": r["instructions"], "image": r.get("image",""),
        "tags": r.get("tags",""), "ingredients": r["ingredients"],
        "measures": r.get("measures",[]),
        "youtube": "", "difficulty": r.get("difficulty","Easy"),
        "time_mins": r.get("time_mins"),
    } for r in BONUS_RECIPES]


def compute_coverage(user_ings_set, recipe_ings):
    """
    Compute ingredient coverage with fuzzy matching.
    Returns (coverage_pct, matched_list, missing_list).
    """
    recipe_set = set(i.lower() for i in recipe_ings)
    user_set   = set(i.lower() for i in user_ings_set)
    matched, missing = set(), set()
    for r_ing in recipe_set:
        found = any(u in r_ing or r_ing in u for u in user_set)
        (matched if found else missing).add(r_ing)
    total    = len(recipe_set)
    coverage = (len(matched) / total * 100) if total > 0 else 0
    return round(coverage, 1), list(matched), list(missing)


def match_recipes(user_ingredients, all_recipes,
                  diet_filter="All", cuisine_filter="All",
                  min_coverage=20):
    """
    Match recipes to user ingredients.
    Filter by diet and cuisine.
    Return sorted list with coverage, matched, missing.
    """
    user_set = set(i.strip().lower() for i in user_ingredients if i.strip())
    results  = []
    for recipe in all_recipes:
        ings = recipe.get("ingredients", [])
        if not ings:
            continue
        coverage, matched, missing = compute_coverage(user_set, ings)
        if coverage < min_coverage:
            continue

        tags = str(recipe.get("tags") or "").lower()
        cat  = str(recipe.get("category") or "").lower()

        if diet_filter == "Vegetarian":
            if cat in ["beef","chicken","lamb","pork","seafood"]:
                continue
        if diet_filter == "Vegan":
            if cat in ["beef","chicken","lamb","pork","seafood","dairy"]:
                continue
            if not "vegan" in tags:
                # basic check: skip if contains obvious animal products
                if any(a in " ".join(ings) for a in
                       ["meat","chicken","beef","pork","fish","egg",
                        "milk","cheese","butter","cream"]):
                    continue

        if cuisine_filter != "All":
            area = str(recipe.get("area") or "").lower()
            if cuisine_filter.lower() not in area:
                continue

        allergens = detect_allergens(ings)
        results.append({
            **recipe,
            "coverage":      coverage,
            "matched":       matched,
            "missing":       missing,
            "missing_count": len(missing),
            "allergens":     allergens,
        })

    results.sort(key=lambda x: (-x["coverage"], x["missing_count"]))
    return results


# ─────────────────────────────────────────────
# SERVING SIZE SCALER
# ─────────────────────────────────────────────

def scale_measure(measure_str, factor):
    """
    Scale a measure string by a factor.
    e.g. "200g" × 2 = "400g"
         "3 cloves" × 1.5 = "4.5 cloves"
    Handles common units: g, kg, ml, l, tbsp, tsp, cups.
    """
    if not measure_str or measure_str.strip() in ["to taste","as needed",""]:
        return measure_str

    import re
    # Find leading number
    match = re.match(r"^([\d.\/]+)\s*(.*)", measure_str.strip())
    if not match:
        return measure_str

    num_str, unit = match.group(1), match.group(2)

    # Handle fractions like 1/2
    try:
        if "/" in num_str:
            parts = num_str.split("/")
            num = float(parts[0]) / float(parts[1])
        else:
            num = float(num_str)
    except ValueError:
        return measure_str

    scaled = num * factor

    # Format nicely
    if scaled == int(scaled):
        scaled_str = str(int(scaled))
    else:
        scaled_str = f"{scaled:.1f}"

    return f"{scaled_str} {unit}".strip()


def display_scaled_ingredients(recipe, servings, base_servings=4):
    """Display ingredient list scaled to chosen serving size."""
    factor = servings / base_servings
    ings   = recipe.get("ingredients", [])
    msrs   = recipe.get("measures", [])

    rows = []
    for i, ing in enumerate(ings):
        msr = msrs[i] if i < len(msrs) else ""
        scaled = scale_measure(msr, factor)
        rows.append({"Ingredient": ing.title(), "Quantity": scaled})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ML: KNN RECOMMENDATION
# ─────────────────────────────────────────────

def recommend_by_knn(favourites_names, all_recipes,
                     user_ingredients=None, n=5):
    """
    KNN recipe recommendation using ingredient profile similarity.
    If no favourites, falls back to user's current ingredients.

    Each recipe = binary vector (1 if ingredient present, 0 if not).
    Cosine distance finds most similar recipes.
    """
    if len(all_recipes) < 5:
        return []

    all_ings = sorted(set(
        ing for r in all_recipes for ing in r.get("ingredients",[])
    ))
    if not all_ings:
        return []

    mlb = MultiLabelBinarizer(classes=all_ings)
    X   = mlb.fit_transform([r.get("ingredients",[]) for r in all_recipes])

    # Build query: average of favourites OR current ingredients
    if favourites_names:
        fav_idx = [i for i, r in enumerate(all_recipes)
                   if r["name"] in favourites_names]
        if not fav_idx:
            return []
        query = X[fav_idx].mean(axis=0).reshape(1,-1)
        exclude = set(favourites_names)
    elif user_ingredients:
        # Fall back: use user's ingredients as query
        user_vec = mlb.transform([
            [i for i in user_ingredients if i in all_ings]
        ])
        query   = user_vec.astype(float)
        exclude = set()
    else:
        return []

    knn = NearestNeighbors(
        n_neighbors=min(n + len(exclude), len(all_recipes)),
        metric="cosine", algorithm="brute"
    )
    knn.fit(X)
    distances, indices = knn.kneighbors(query)

    recs = []
    for idx, dist in zip(indices[0], distances[0]):
        r = all_recipes[idx]
        if r["name"] not in exclude:
            recs.append({**r, "similarity": round((1-dist)*100, 1)})
        if len(recs) >= n:
            break
    return recs


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_coverage_bar(results):
    top = results[:15]
    colors = ["#1D9E75" if c["coverage"] >= 80
              else ("#f0c040" if c["coverage"] >= 50 else "#E8734A")
              for c in top]
    fig = go.Figure(go.Bar(
        x=[r["coverage"] for r in top],
        y=[r["name"][:30] for r in top],
        orientation="h",
        marker_color=colors,
        text=[f"{r['coverage']:.0f}%" for r in top],
        textposition="auto"
    ))
    fig.update_layout(
        title="Ingredient coverage — top matches",
        xaxis=dict(range=[0,100], title="% ingredients you have"),
        height=max(300, len(top)*28),
        margin=dict(l=10,r=10,t=40,b=10),
        yaxis=dict(autorange="reversed")
    )
    return fig




def plot_category_donut(results):
    cats = {}
    for r in results:
        c = str(r.get("category") or "Other")
        cats[c] = cats.get(c, 0) + 1
    if not cats:
        return None
    fig = px.pie(
        values=list(cats.values()), names=list(cats.keys()),
        hole=0.45, title="Recipe categories",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
    return fig


def plot_missing_bar(results):
    miss = {}
    for r in results[:20]:
        for ing in r.get("missing", []):
            miss[ing] = miss.get(ing, 0) + 1
    if not miss:
        return None
    top = sorted(miss.items(), key=lambda x: -x[1])[:10]
    fig = px.bar(
        x=[t[1] for t in top], y=[t[0] for t in top],
        orientation="h",
        labels={"x":"Recipes needing it","y":""},
        title="Most needed missing ingredients",
        color=[t[1] for t in top],
        color_continuous_scale="Reds"
    )
    fig.update_layout(
        height=320, showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10,r=10,t=40,b=10),
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_allergen_summary(results):
    """Bar chart of allergen frequency across matched recipes."""
    allergen_count = {}
    for r in results:
        for a in r.get("allergens", []):
            allergen_count[a] = allergen_count.get(a, 0) + 1
    if not allergen_count:
        return None
    sorted_a = sorted(allergen_count.items(), key=lambda x: -x[1])
    fig = px.bar(
        x=[a[0] for a in sorted_a],
        y=[a[1] for a in sorted_a],
        labels={"x":"Allergen","y":"Number of recipes"},
        title="Allergen frequency in your results",
        color=[a[1] for a in sorted_a],
        color_continuous_scale="Oranges"
    )
    fig.update_layout(
        height=300, showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis_tickangle=-20
    )
    return fig


# ─────────────────────────────────────────────
# RECIPE CARD DISPLAY
# ─────────────────────────────────────────────

# Global button counter stored in session state
def next_key(prefix):
    """Generate a unique key for Streamlit widgets."""
    k = f"_key_ctr_{prefix}"
    if k not in st.session_state:
        st.session_state[k] = 0
    st.session_state[k] += 1
    return f"{prefix}_{st.session_state[k]}"


def display_recipe_card(recipe, user_ings, servings=4,
                        show_save=True, key_prefix="card"):
    """
    Full recipe card with:
    - Ingredient coverage (matched/missing)
    - Scaled ingredient list
    - Allergen warnings
    - Nutrition info (if Edamam key provided)
    - Instructions
    - Save button
    """
    coverage = recipe.get("coverage", 0)
    matched  = recipe.get("matched", [])
    missing  = recipe.get("missing", [])
    allergens= recipe.get("allergens", [])

    cov_color = "🟢" if coverage >= 80 else ("🟡" if coverage >= 50 else "🔴")

    with st.container():
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown(f"### {recipe['name']}")
            badges = []
            if recipe.get("area"):
                badges.append(f"🌍 {recipe['area']}")
            if recipe.get("category"):
                badges.append(f"🍽️ {recipe['category']}")
            if recipe.get("difficulty"):
                badges.append(f"⚡ {recipe['difficulty']}")
            if recipe.get("time_mins"):
                badges.append(f"⏱️ {recipe['time_mins']} min")
            if recipe.get("source") == "hardcoded":
                badges.append("🇨🇭 Bonus recipe")
            st.markdown("  ".join(badges))

            st.markdown(
                f"{cov_color} **{coverage:.0f}% covered** "
                f"({len(matched)}/{len(matched)+len(missing)} ingredients)"
            )

            # Matched
            if matched:
                html = " ".join(
                    f'<span class="ingredient-tag">✅ {i}</span>'
                    for i in sorted(matched))
                st.markdown(f"**You have:** {html}",
                            unsafe_allow_html=True)
            # Missing
            if missing:
                html = " ".join(
                    f'<span class="missing-tag">❌ {i}</span>'
                    for i in sorted(missing))
                st.markdown(f"**Still need:** {html}",
                            unsafe_allow_html=True)

            # Allergens
            if allergens:
                html = " ".join(
                    f'<span class="allergen-tag">{a}</span>'
                    for a in allergens)
                st.markdown(f"**⚠️ Allergens:** {html}",
                            unsafe_allow_html=True)
            else:
                st.markdown("✅ **No major allergens detected**")

        with c2:
            if recipe.get("image"):
                st.image(recipe["image"], use_container_width=True)
            if show_save:
                if st.button("❤️ Save", key=next_key(key_prefix)):
                    save_favourite(recipe["id"], recipe["name"],
                                   recipe.get("source","api"))
                    st.success("Saved!")

        # Expandable details
        with st.expander("🥄 Ingredients & Quantities"):
            st.markdown(f"*Scaled for {servings} servings*")
            display_scaled_ingredients(recipe, servings)

        with st.expander("📖 Instructions"):
            instructions = recipe.get("instructions","")
            if instructions:
                st.markdown(instructions)
            if recipe.get("youtube"):
                st.markdown(f"[▶️ Watch on YouTube]({recipe['youtube']})")


        st.divider()


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

def main():
    init_db()

    st.title("🍳 Smart Cookbook")
    st.markdown(
        "Enter your ingredients → get matching recipes with "
        "**quantities, allergens, nutrition** and **serving scaler**."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("🥕 Your Ingredients")
        ingredients_text = st.text_area(
            "Ingredients (one per line)",
            placeholder="chicken\nonion\ngarlic\ntomatoes\nolive oil",
            height=160, label_visibility="collapsed"
        )

        st.markdown("**Quick add:**")
        quick_list = ["chicken", "eggs", "pasta", "rice", "potatoes",
                      "onion", "garlic", "tomatoes", "cheese", "butter",
                      "milk", "flour", "beef", "salmon", "mushrooms"]
        quick_selected = []
        qc1, qc2 = st.columns(2)
        for i, ing in enumerate(quick_list):
            col = qc1 if i % 2 == 0 else qc2
            if col.checkbox(ing, key=f"q_{ing}"):
                quick_selected.append(ing)

        st.divider()
        st.header("🔧 Filters")
        diet_filter    = st.selectbox("Diet", ["All","Vegetarian","Vegan"])
        cuisine_filter = st.selectbox("Cuisine", [
            "All","Italian","French","Spanish","Swiss",
            "Asian","American","British","Mexican","Mediterranean"])
        min_coverage   = st.slider("Min coverage %", 0, 100, 30, 5)
        servings       = st.number_input("👨‍👩‍👧 Servings", 1, 12, 4, 1)
        include_bonus  = st.checkbox("Include Swiss/European recipes", True)



        st.divider()
        st.header("⚠️ Allergen Filter")
        allergen_exclude = st.multiselect(
            "Exclude recipes containing:",
            list(ALLERGEN_KEYWORDS.keys())
        )

        search_btn = st.button("🔍 Find Recipes!",
                               type="primary", use_container_width=True)

    # ── Parse ingredients ──
    typed = [i.strip().lower() for i in
             (ingredients_text or "").split("\n") if i.strip()]
    user_ingredients = list(set(typed + quick_selected))

    if not user_ingredients:
        st.info("👈 Enter ingredients in the sidebar and click **Find Recipes!**")
        st.subheader("🎲 Recipe of the day")
        with st.spinner("Loading..."):
            rand = api_get_random()
        if rand:
            rc1, rc2 = st.columns([2,1])
            with rc1:
                st.markdown(f"### {rand['strMeal']}")
                st.markdown(
                    f"🌍 {rand.get('strArea','')} | "
                    f"🍽️ {rand.get('strCategory','')}"
                )
                allergens = detect_allergens([
                    (rand.get(f"strIngredient{i}") or "").lower()
                    for i in range(1,21)])
                if allergens:
                    st.warning("⚠️ Allergens: " + ", ".join(allergens))
                with st.expander("📖 Instructions"):
                    st.markdown(rand.get("strInstructions",""))
            with rc2:
                if rand.get("strMealThumb"):
                    st.image(rand["strMealThumb"],
                             use_container_width=True)
        return

    # ── Fetch and match ──
    with st.spinner("Searching recipes..."):
        api_recipes  = fetch_recipes_for_ingredients(user_ingredients)
        bonus        = get_bonus_recipes() if include_bonus else []
        all_recipes  = api_recipes + bonus
        matched_recipes = match_recipes(
            user_ingredients, all_recipes,
            diet_filter, cuisine_filter, min_coverage
        )

    # Apply allergen exclusion filter
    if allergen_exclude:
        matched_recipes = [
            r for r in matched_recipes
            if not any(a in r.get("allergens",[]) for a in allergen_exclude)
        ]

    # Log search
    top_name = matched_recipes[0]["name"] if matched_recipes else "None"
    log_search(user_ingredients, diet_filter, len(matched_recipes), top_name)

    # ── Find "1 ingredient away" recipes ──
    one_away = [r for r in match_recipes(
        user_ingredients, all_recipes, diet_filter, cuisine_filter, 0)
        if r["missing_count"] == 1 and r not in matched_recipes]

    # ── Tabs ──
    (tab_results, tab_one_away, tab_analytics,
     tab_ml, tab_favs, tab_rate, tab_log) = st.tabs([
        "🍽️ Recipes",
        "1️⃣ 1 Ingredient Away",
        "📊 Analytics",
        "🤖 ML Recommendations",
        "❤️ Favourites",
        "⭐ Rate",
        "🕒 History"
    ])

    # ── TAB 1: RESULTS ──
    with tab_results:
        st.subheader(
            f"🍽️ {len(matched_recipes)} recipes for: "
            f"{', '.join(user_ingredients[:5])}"
            + ("..." if len(user_ingredients) > 5 else "")
        )

        if not matched_recipes:
            st.warning("No recipes found. Try lowering coverage % or "
                       "adding more ingredients.")
        else:
            qs1,qs2,qs3,qs4 = st.columns(4)
            can_now = [r for r in matched_recipes if r["coverage"] >= 80]
            qs1.metric("Total matches",   len(matched_recipes))
            qs2.metric("Can make NOW 🟢", len(can_now))
            qs3.metric("Best coverage",
                       f"{matched_recipes[0]['coverage']:.0f}%")
            qs4.metric("Your ingredients", len(user_ingredients))

            sort_by = st.radio(
                "Sort by:",
                ["Coverage","Fewest missing","Alphabetical"],
                horizontal=True
            )
            if sort_by == "Fewest missing":
                matched_recipes = sorted(
                    matched_recipes, key=lambda x: x["missing_count"])
            elif sort_by == "Alphabetical":
                matched_recipes = sorted(
                    matched_recipes, key=lambda x: x["name"])

            for recipe in matched_recipes:
                display_recipe_card(
                    recipe, user_ingredients, servings,
                    key_prefix=f"res_{recipe['id']}"
                )

    # ── TAB 2: 1 INGREDIENT AWAY ──
    with tab_one_away:
        st.subheader("1️⃣ Just 1 Ingredient Away!")
        st.markdown(
            "These recipes need only **one more ingredient** "
            "you don't have. Worth a quick shopping trip!"
        )
        if not one_away:
            st.info("No recipes found that need exactly 1 more ingredient.")
        else:
            for recipe in one_away[:10]:
                missing_ing = recipe["missing"][0] if recipe["missing"] else "?"
                st.markdown(
                    f"**{recipe['name']}** — just need: "
                    f"**{missing_ing}** | "
                    f"Coverage: {recipe['coverage']:.0f}%"
                )
                display_recipe_card(
                    recipe, user_ingredients, servings,
                    key_prefix=f"away_{recipe['id']}"
                )

    # ── TAB 3: ANALYTICS ──
    with tab_analytics:
        st.subheader("📊 Analytics")
        if not matched_recipes:
            st.info("Run a search first.")
        else:
            ac1,ac2 = st.columns(2)
            with ac1:
                st.plotly_chart(plot_coverage_bar(matched_recipes),
                                use_container_width=True, key="cov_bar")
            with ac2:
                fig_cat = plot_category_donut(matched_recipes)
                if fig_cat:
                    st.plotly_chart(fig_cat, use_container_width=True,
                                    key="cat_donut")

            ac3,ac4 = st.columns(2)
            with ac3:
                fig_miss = plot_missing_bar(matched_recipes)
                if fig_miss:
                    st.plotly_chart(fig_miss, use_container_width=True,
                                    key="miss_bar")
            with ac4:
                fig_allerg = plot_allergen_summary(matched_recipes)
                if fig_allerg:
                    st.plotly_chart(fig_allerg, use_container_width=True,
                                    key="allerg_bar")

            st.divider()
            st.subheader("🛒 Shopping List")
            all_missing = set()
            for r in matched_recipes:
                all_missing.update(r.get("missing",[]))
            user_set = set(i.lower() for i in user_ingredients)
            truly_missing = sorted([
                i for i in all_missing
                if not any(u in i or i in u for u in user_set)
            ])
            if truly_missing:
                cols = st.columns(3)
                for i, ing in enumerate(truly_missing):
                    cols[i%3].markdown(f"- {ing}")
            else:
                st.success("You have everything! 🎉")

    # ── TAB 4: ML RECOMMENDATIONS ──
    with tab_ml:
        st.subheader("🤖 KNN Recipe Recommendations")
        st.markdown(
            "Our **K-Nearest Neighbors** model finds recipes with "
            "similar ingredient profiles. Saves first, then personalises."
        )
        df_favs_ml = load_favourites()
        fav_names  = df_favs_ml["recipe_name"].tolist() \
            if not df_favs_ml.empty else []

        if fav_names:
            st.markdown(
                f"Based on your {len(fav_names)} favourites: "
                f"**{', '.join(fav_names[:4])}**"
                + ("..." if len(fav_names) > 4 else "")
            )
        else:
            st.info(
                "No saved favourites yet — showing recommendations "
                "based on your current ingredients instead."
            )

        with st.spinner("Running KNN..."):
            recs = recommend_by_knn(
                fav_names, all_recipes, user_ingredients, n=6)

        if not recs:
            st.info("Not enough data yet. Add more ingredients or save favourites.")
        else:
            for rec in recs:
                st.markdown(
                    f"**{rec['name']}** — "
                    f"{rec.get('area','')} | "
                    f"Similarity: {rec.get('similarity',0):.0f}%"
                )
                # compute coverage for display
                cov, mat, mis = compute_coverage(
                    set(user_ingredients), rec.get("ingredients",[]))
                rec["coverage"] = cov
                rec["matched"]  = mat
                rec["missing"]  = mis
                rec["missing_count"] = len(mis)
                rec["allergens"] = detect_allergens(
                    rec.get("ingredients",[]))
                display_recipe_card(
                    rec, user_ingredients, servings,
                    key_prefix=f"ml_{rec['id']}"
                )

        st.markdown("""
        ---
        💡 **How it works:** Each recipe is a binary vector where
        1 = ingredient present. KNN finds recipes whose vectors
        are most similar (smallest cosine distance) to your
        favourites' average vector. No manual rules — pure
        ingredient similarity.
        """)

    # ── TAB 5: FAVOURITES ──
    with tab_favs:
        st.subheader("❤️ Saved Recipes")
        df_favs_tab = load_favourites()
        if df_favs_tab.empty:
            st.info("No saved recipes yet. Click ❤️ Save on any recipe!")
        else:
            st.dataframe(
                df_favs_tab[["recipe_name","source","saved_at"]].rename(
                    columns={"recipe_name":"Recipe",
                             "source":"Source","saved_at":"Saved at"}),
                use_container_width=True, hide_index=True
            )
            if st.button("🗑️ Clear favourites"):
                conn = sqlite3.connect("cookbook_v2.db")
                conn.execute("DELETE FROM favourites")
                conn.commit()
                conn.close()
                st.rerun()

    # ── TAB 6: RATE ──
    with tab_rate:
        st.subheader("⭐ Rate Recipes")
        if not matched_recipes:
            st.info("Search for recipes first.")
        else:
            to_rate = st.selectbox(
                "Which recipe did you try?",
                [r["name"] for r in matched_recipes]
            )
            rating = st.slider("Rating", 1, 5, 3)
            st.markdown("⭐" * rating)
            notes = st.text_input("Notes",
                                  placeholder="Delicious! Added extra garlic...")
            if st.button("✅ Submit"):
                rid = next(
                    (r["id"] for r in matched_recipes if r["name"] == to_rate),
                    to_rate)
                save_rating(rid, to_rate, rating, notes)
                st.success(f"Rated: {'⭐'*rating}")

            df_rat = load_ratings()
            if not df_rat.empty:
                st.divider()
                df_rat["stars"] = df_rat["rating"].apply(
                    lambda r: "⭐"*int(r))
                st.dataframe(
                    df_rat[["recipe_name","stars","notes","rated_at"]].rename(
                        columns={"recipe_name":"Recipe","stars":"Rating",
                                 "notes":"Notes","rated_at":"Date"}),
                    use_container_width=True, hide_index=True
                )

    # ── TAB 7: HISTORY ──
    with tab_log:
        st.subheader("🕒 Search History")
        df_log = load_search_log()
        if df_log.empty:
            st.info("No searches yet.")
        else:
            st.dataframe(df_log, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear history"):
            conn = sqlite3.connect("cookbook_v2.db")
            conn.execute("DELETE FROM search_log")
            conn.commit()
            conn.close()
            st.rerun()


if __name__ == "__main__":
    main()
