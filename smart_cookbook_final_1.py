"""
🍳 Smart Cookbook — What Can I Cook?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enter the ingredients you have at home and discover
what you can cook. Combines live API recipes with
handcrafted Swiss/European recipes.

APIs (all free, no key required):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TheMealDB  → 600+ real recipes, ingredients, instructions, photos
   https://www.themealdb.com/api/json/v1/1/

ML component:
━━━━━━━━━━━━
- KNN similarity matching: finds recipes whose ingredient
  profiles are most similar to what the user has at home
- Ingredient coverage score: how many % of a recipe's
  ingredients you already have

Hardcoded:
━━━━━━━━━━
- 15 Swiss/European bonus recipes (not in TheMealDB)
- Difficulty ratings and cook time estimates

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
    .recipe-card {
        background: linear-gradient(135deg, #fff9f0, #fff3e0);
        border: 1px solid #f0a500;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HARDCODED SWISS/EUROPEAN BONUS RECIPES
# These supplement the API with local recipes
# not available in TheMealDB.
# ─────────────────────────────────────────────

BONUS_RECIPES = [
    {
        "id": "swiss_001",
        "name": "Rösti",
        "category": "Side",
        "area": "Swiss",
        "difficulty": "Easy",
        "time_mins": 30,
        "ingredients": ["potatoes", "butter", "salt", "pepper", "onion"],
        "instructions": (
            "1. Grate potatoes coarsely and squeeze out excess water.\n"
            "2. Season with salt and pepper, mix in grated onion.\n"
            "3. Heat butter in a non-stick pan over medium heat.\n"
            "4. Press potato mixture into pan to form a flat cake.\n"
            "5. Cook 10-12 min per side until golden and crispy.\n"
            "6. Slide onto plate and serve immediately."
        ),
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/thirty-3/R%C3%B6sti_2.jpg/320px-R%C3%B6sti_2.jpg",
        "tags": "Swiss,Potato,Vegetarian",
    },
    {
        "id": "swiss_002",
        "name": "Cheese Fondue",
        "category": "Dinner",
        "area": "Swiss",
        "difficulty": "Medium",
        "time_mins": 25,
        "ingredients": ["gruyere cheese", "emmental cheese", "white wine",
                        "garlic", "bread", "cornstarch", "kirsch", "nutmeg"],
        "instructions": (
            "1. Rub fondue pot with halved garlic clove.\n"
            "2. Pour wine into pot and heat gently.\n"
            "3. Gradually add grated cheese, stirring in a figure-8 motion.\n"
            "4. Mix cornstarch with kirsch, add to cheese mixture.\n"
            "5. Season with nutmeg and pepper.\n"
            "6. Keep warm over low heat. Dip bread cubes to serve."
        ),
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Swiss_cheese_fondue.jpg/320px-Swiss_cheese_fondue.jpg",
        "tags": "Swiss,Cheese,Dinner Party",
    },
    {
        "id": "swiss_003",
        "name": "Zürcher Geschnetzeltes",
        "category": "Dinner",
        "area": "Swiss",
        "difficulty": "Medium",
        "time_mins": 35,
        "ingredients": ["veal", "mushrooms", "cream", "white wine",
                        "onion", "butter", "lemon", "parsley"],
        "instructions": (
            "1. Cut veal into thin strips, season with salt and pepper.\n"
            "2. Sauté onions in butter until soft.\n"
            "3. Add veal strips, cook quickly over high heat 2-3 min.\n"
            "4. Add sliced mushrooms, cook 3 more minutes.\n"
            "5. Deglaze with white wine, reduce by half.\n"
            "6. Add cream, simmer 5 minutes until slightly thickened.\n"
            "7. Finish with lemon juice and fresh parsley.\n"
            "8. Serve with Rösti or egg noodles."
        ),
        "image": "",
        "tags": "Swiss,Veal,Zurich",
    },
    {
        "id": "swiss_004",
        "name": "Birchermüesli",
        "category": "Breakfast",
        "area": "Swiss",
        "difficulty": "Easy",
        "time_mins": 10,
        "ingredients": ["oats", "milk", "apple", "lemon juice",
                        "honey", "yogurt", "nuts", "berries"],
        "instructions": (
            "1. Soak oats in milk overnight in the fridge.\n"
            "2. Grate apple and mix with lemon juice.\n"
            "3. Combine oats, grated apple, yogurt and honey.\n"
            "4. Top with mixed berries and crushed nuts.\n"
            "5. Serve cold."
        ),
        "image": "",
        "tags": "Swiss,Breakfast,Healthy,Vegetarian",
    },
    {
        "id": "swiss_005",
        "name": "Raclette",
        "category": "Dinner",
        "area": "Swiss",
        "difficulty": "Easy",
        "time_mins": 20,
        "ingredients": ["raclette cheese", "potatoes", "pickles",
                        "pearl onions", "pepper"],
        "instructions": (
            "1. Boil potatoes in their skins until tender.\n"
            "2. Melt raclette cheese under grill or raclette machine.\n"
            "3. Scrape melted cheese onto potatoes.\n"
            "4. Serve with pickles and pearl onions.\n"
            "5. Season with freshly ground pepper."
        ),
        "image": "",
        "tags": "Swiss,Cheese,Winter",
    },
    {
        "id": "eur_001",
        "name": "Pasta Carbonara",
        "category": "Dinner",
        "area": "Italian",
        "difficulty": "Medium",
        "time_mins": 20,
        "ingredients": ["spaghetti", "eggs", "pancetta", "parmesan",
                        "black pepper", "salt", "garlic"],
        "instructions": (
            "1. Cook spaghetti in salted water until al dente.\n"
            "2. Fry pancetta with garlic until crispy.\n"
            "3. Mix eggs with grated parmesan and black pepper.\n"
            "4. Remove pan from heat, add drained pasta.\n"
            "5. Pour egg mixture over, toss quickly — residual heat cooks eggs.\n"
            "6. Add pasta water to loosen if needed. Serve immediately."
        ),
        "image": "",
        "tags": "Italian,Pasta,Quick",
    },
    {
        "id": "eur_002",
        "name": "French Onion Soup",
        "category": "Starter",
        "area": "French",
        "difficulty": "Medium",
        "time_mins": 60,
        "ingredients": ["onion", "beef broth", "butter", "white wine",
                        "bread", "gruyere cheese", "thyme", "bay leaf"],
        "instructions": (
            "1. Slice onions thinly. Caramelise in butter over low heat 40 min.\n"
            "2. Add white wine, thyme, bay leaf. Cook 5 minutes.\n"
            "3. Add beef broth, simmer 15 minutes. Season.\n"
            "4. Ladle into oven-safe bowls.\n"
            "5. Top with toasted bread slices and grated gruyère.\n"
            "6. Grill until cheese is bubbly and golden."
        ),
        "image": "",
        "tags": "French,Soup,Winter",
    },
    {
        "id": "eur_003",
        "name": "Spanish Omelette (Tortilla)",
        "category": "Breakfast",
        "area": "Spanish",
        "difficulty": "Easy",
        "time_mins": 30,
        "ingredients": ["eggs", "potatoes", "onion", "olive oil", "salt"],
        "instructions": (
            "1. Slice potatoes and onions thinly.\n"
            "2. Fry slowly in generous olive oil until soft (not crispy).\n"
            "3. Beat eggs with salt in a bowl.\n"
            "4. Drain potato mixture, add to eggs.\n"
            "5. Cook in oiled pan until bottom is set.\n"
            "6. Flip using a plate. Cook other side 3 minutes.\n"
            "7. Serve warm or at room temperature."
        ),
        "image": "",
        "tags": "Spanish,Eggs,Vegetarian",
    },
    {
        "id": "eur_004",
        "name": "Beef Stroganoff",
        "category": "Dinner",
        "area": "Russian",
        "difficulty": "Medium",
        "time_mins": 40,
        "ingredients": ["beef", "mushrooms", "sour cream", "onion",
                        "butter", "beef broth", "mustard", "paprika"],
        "instructions": (
            "1. Cut beef into thin strips, season.\n"
            "2. Sauté onions until golden. Set aside.\n"
            "3. Brown beef quickly in batches. Set aside.\n"
            "4. Cook mushrooms in same pan.\n"
            "5. Return beef and onions. Add broth, mustard, paprika.\n"
            "6. Simmer 5 minutes. Stir in sour cream.\n"
            "7. Serve over egg noodles or rice."
        ),
        "image": "",
        "tags": "Russian,Beef,Comfort Food",
    },
    {
        "id": "eur_005",
        "name": "Minestrone Soup",
        "category": "Starter",
        "area": "Italian",
        "difficulty": "Easy",
        "time_mins": 45,
        "ingredients": ["tomatoes", "carrot", "celery", "onion", "garlic",
                        "zucchini", "pasta", "vegetable broth",
                        "olive oil", "parmesan", "basil"],
        "instructions": (
            "1. Sauté onion, carrot, celery in olive oil 5 min.\n"
            "2. Add garlic and tomatoes. Cook 3 minutes.\n"
            "3. Pour in broth, bring to boil.\n"
            "4. Add zucchini and pasta. Simmer 12 minutes.\n"
            "5. Season with salt, pepper, fresh basil.\n"
            "6. Serve with grated parmesan."
        ),
        "image": "",
        "tags": "Italian,Soup,Vegetarian",
    },
    {
        "id": "eur_006",
        "name": "Shakshuka",
        "category": "Breakfast",
        "area": "Middle Eastern",
        "difficulty": "Easy",
        "time_mins": 25,
        "ingredients": ["eggs", "tomatoes", "bell pepper", "onion",
                        "garlic", "cumin", "paprika", "olive oil", "feta cheese"],
        "instructions": (
            "1. Sauté onion and pepper in olive oil until soft.\n"
            "2. Add garlic, cumin, paprika. Cook 1 minute.\n"
            "3. Add crushed tomatoes, simmer 10 minutes.\n"
            "4. Make wells in sauce, crack eggs in.\n"
            "5. Cover, cook 5-8 minutes until whites are set.\n"
            "6. Crumble feta on top. Serve with bread."
        ),
        "image": "",
        "tags": "Middle Eastern,Eggs,Vegetarian,Breakfast",
    },
    {
        "id": "eur_007",
        "name": "Lentil Soup",
        "category": "Starter",
        "area": "Mediterranean",
        "difficulty": "Easy",
        "time_mins": 40,
        "ingredients": ["lentils", "onion", "carrot", "garlic", "cumin",
                        "turmeric", "olive oil", "lemon", "vegetable broth"],
        "instructions": (
            "1. Sauté onion and carrot in olive oil until soft.\n"
            "2. Add garlic, cumin, turmeric. Cook 1 minute.\n"
            "3. Add rinsed lentils and broth. Bring to boil.\n"
            "4. Simmer 25 minutes until lentils are very soft.\n"
            "5. Blend partially for creamy texture.\n"
            "6. Finish with lemon juice and olive oil drizzle."
        ),
        "image": "",
        "tags": "Mediterranean,Vegan,Healthy",
    },
    {
        "id": "eur_008",
        "name": "Banana Pancakes",
        "category": "Breakfast",
        "area": "American",
        "difficulty": "Easy",
        "time_mins": 15,
        "ingredients": ["banana", "eggs", "flour", "milk",
                        "baking powder", "butter", "honey"],
        "instructions": (
            "1. Mash banana in a bowl.\n"
            "2. Beat in eggs, then add milk.\n"
            "3. Stir in flour and baking powder until smooth.\n"
            "4. Heat butter in pan over medium heat.\n"
            "5. Pour small ladles of batter, cook 2 min per side.\n"
            "6. Serve with honey or maple syrup."
        ),
        "image": "",
        "tags": "Breakfast,Sweet,Easy",
    },
    {
        "id": "eur_009",
        "name": "Garlic Butter Shrimp Pasta",
        "category": "Dinner",
        "area": "Italian",
        "difficulty": "Easy",
        "time_mins": 20,
        "ingredients": ["shrimp", "pasta", "garlic", "butter",
                        "white wine", "parsley", "lemon", "chili flakes"],
        "instructions": (
            "1. Cook pasta in salted water until al dente.\n"
            "2. Melt butter in pan, sauté garlic 1 minute.\n"
            "3. Add shrimp, cook 2 min per side until pink.\n"
            "4. Deglaze with white wine, cook 2 minutes.\n"
            "5. Add pasta, toss with lemon juice and parsley.\n"
            "6. Finish with chili flakes and more butter."
        ),
        "image": "",
        "tags": "Italian,Seafood,Quick",
    },
    {
        "id": "eur_010",
        "name": "Vegetable Stir Fry",
        "category": "Dinner",
        "area": "Asian",
        "difficulty": "Easy",
        "time_mins": 15,
        "ingredients": ["broccoli", "carrot", "bell pepper", "garlic",
                        "soy sauce", "sesame oil", "ginger",
                        "rice", "cornstarch"],
        "instructions": (
            "1. Cook rice according to packet instructions.\n"
            "2. Heat sesame oil in wok over high heat.\n"
            "3. Add garlic and ginger, stir 30 seconds.\n"
            "4. Add vegetables, stir fry 4-5 minutes.\n"
            "5. Mix soy sauce with cornstarch, pour over veg.\n"
            "6. Toss until sauce coats everything. Serve over rice."
        ),
        "image": "",
        "tags": "Asian,Vegan,Quick,Healthy",
    },
]


# ─────────────────────────────────────────────
# DATABASE
# Tables:
#   api_recipes      - cached recipes from TheMealDB
#   favourites       - user saved recipes
#   search_log       - every ingredient search logged
#   user_ratings     - user ratings for recipes
# ─────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect("cookbook.db")
    c = conn.cursor()

    # Cached API recipes
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_recipes (
            meal_id     TEXT PRIMARY KEY,
            name        TEXT,
            category    TEXT,
            area        TEXT,
            instructions TEXT,
            image_url   TEXT,
            tags        TEXT,
            ingredients TEXT,
            youtube_url TEXT,
            fetched_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # User favourite recipes
    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id   TEXT,
            recipe_name TEXT,
            source      TEXT,
            saved_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Every ingredient search logged
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
            ingredients TEXT,
            diet_filter TEXT,
            cuisine_filter TEXT,
            results_count INTEGER,
            top_result  TEXT
        )
    """)

    # User ratings
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            recipe_id   TEXT PRIMARY KEY,
            recipe_name TEXT,
            rating      INTEGER,
            notes       TEXT,
            rated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_api_recipe(meal):
    """Parse TheMealDB response and save to database."""
    # Extract ingredients (TheMealDB stores them as ingredient1..20, measure1..20)
    ingredients = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        if ing:
            ingredients.append(ing.lower())

    conn = sqlite3.connect("cookbook.db")
    conn.execute("""
        INSERT OR REPLACE INTO api_recipes VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
    """, (
        meal["idMeal"],
        meal["strMeal"],
        meal.get("strCategory", ""),
        meal.get("strArea", ""),
        meal.get("strInstructions", ""),
        meal.get("strMealThumb", ""),
        meal.get("strTags", ""),
        "|".join(ingredients),
        meal.get("strYoutube", ""),
    ))
    conn.commit()
    conn.close()


def load_cached_recipes():
    """Load all cached API recipes from database."""
    conn = sqlite3.connect("cookbook.db")
    df = pd.read_sql_query("SELECT * FROM api_recipes", conn)
    conn.close()
    return df


def save_favourite(recipe_id, recipe_name, source):
    conn = sqlite3.connect("cookbook.db")
    conn.execute("""
        INSERT INTO favourites (recipe_id, recipe_name, source)
        VALUES (?,?,?)
    """, (recipe_id, recipe_name, source))
    conn.commit()
    conn.close()


def load_favourites():
    conn = sqlite3.connect("cookbook.db")
    df = pd.read_sql_query(
        "SELECT * FROM favourites ORDER BY saved_at DESC", conn)
    conn.close()
    return df


def save_rating(recipe_id, recipe_name, rating, notes):
    conn = sqlite3.connect("cookbook.db")
    conn.execute("""
        INSERT OR REPLACE INTO user_ratings VALUES (?,?,?,?,CURRENT_TIMESTAMP)
    """, (recipe_id, recipe_name, rating, notes))
    conn.commit()
    conn.close()


def load_ratings():
    conn = sqlite3.connect("cookbook.db")
    df = pd.read_sql_query(
        "SELECT * FROM user_ratings ORDER BY rated_at DESC", conn)
    conn.close()
    return df


def log_search(ingredients, diet, cuisine, count, top):
    conn = sqlite3.connect("cookbook.db")
    conn.execute("""
        INSERT INTO search_log
        (ingredients, diet_filter, cuisine_filter, results_count, top_result)
        VALUES (?,?,?,?,?)
    """, (", ".join(ingredients), diet, cuisine, count, top))
    conn.commit()
    conn.close()


def load_search_log():
    conn = sqlite3.connect("cookbook.db")
    df = pd.read_sql_query(
        "SELECT * FROM search_log ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()
    return df


# ─────────────────────────────────────────────
# THEMEALDB API FUNCTIONS
# All free, no key needed.
# Base: https://www.themealdb.com/api/json/v1/1/
# ─────────────────────────────────────────────

BASE = "https://www.themealdb.com/api/json/v1/1"


@st.cache_data(ttl=86400)
def api_search_by_name(query):
    """
    Search recipes by name.
    Endpoint: /search.php?s=query
    Returns list of full meal objects.
    """
    try:
        r = requests.get(f"{BASE}/search.php", params={"s": query}, timeout=10)
        data = r.json()
        return data.get("meals") or []
    except Exception:
        return []


@st.cache_data(ttl=86400)
def api_filter_by_ingredient(ingredient):
    """
    Get all meals containing a specific ingredient.
    Endpoint: /filter.php?i=ingredient
    Returns list of {idMeal, strMeal, strMealThumb}.
    Note: does NOT return full details — need lookup for each.
    """
    try:
        r = requests.get(f"{BASE}/filter.php",
                         params={"i": ingredient}, timeout=10)
        data = r.json()
        return data.get("meals") or []
    except Exception:
        return []


@st.cache_data(ttl=86400)
def api_lookup_by_id(meal_id):
    """
    Get full recipe details by meal ID.
    Endpoint: /lookup.php?i=meal_id
    Returns full meal object with all ingredients and instructions.
    """
    try:
        r = requests.get(f"{BASE}/lookup.php",
                         params={"i": meal_id}, timeout=10)
        data = r.json()
        meals = data.get("meals")
        return meals[0] if meals else None
    except Exception:
        return None


@st.cache_data(ttl=86400)
def api_get_categories():
    """
    Get all meal categories.
    Endpoint: /categories.php
    Returns list of categories with name, image, description.
    """
    try:
        r = requests.get(f"{BASE}/categories.php", timeout=10)
        data = r.json()
        return data.get("categories") or []
    except Exception:
        return []


@st.cache_data(ttl=86400)
def api_get_random():
    """
    Get a random meal.
    Endpoint: /random.php
    """
    try:
        r = requests.get(f"{BASE}/random.php", timeout=10)
        data = r.json()
        meals = data.get("meals")
        return meals[0] if meals else None
    except Exception:
        return None


@st.cache_data(ttl=86400)
def api_filter_by_category(category):
    """
    Get meals by category.
    Endpoint: /filter.php?c=category
    """
    try:
        r = requests.get(f"{BASE}/filter.php",
                         params={"c": category}, timeout=10)
        data = r.json()
        return data.get("meals") or []
    except Exception:
        return []


@st.cache_data(ttl=86400)
def api_filter_by_area(area):
    """
    Get meals by cuisine area.
    Endpoint: /filter.php?a=area
    """
    try:
        r = requests.get(f"{BASE}/filter.php",
                         params={"a": area}, timeout=10)
        data = r.json()
        return data.get("meals") or []
    except Exception:
        return []


@st.cache_data(ttl=86400*7)
def api_list_all_ingredients():
    """
    Get full list of available ingredients.
    Endpoint: /list.php?i=list
    """
    try:
        r = requests.get(f"{BASE}/list.php", params={"i": "list"}, timeout=10)
        data = r.json()
        meals = data.get("meals") or []
        return [m["strIngredient"].lower() for m in meals if m.get("strIngredient")]
    except Exception:
        return []


# ─────────────────────────────────────────────
# RECIPE MATCHING ENGINE
# Core logic: given user's ingredients,
# find best matching recipes.
# ─────────────────────────────────────────────

def parse_ingredients_from_meal(meal_dict):
    """Extract ingredient list from TheMealDB meal object."""
    ingredients = []
    for i in range(1, 21):
        ing = (meal_dict.get(f"strIngredient{i}") or "").strip().lower()
        if ing:
            ingredients.append(ing)
    return ingredients


def compute_coverage(user_ings_set, recipe_ings):
    """
    Compute what % of recipe ingredients the user has.
    Also returns lists of matched and missing ingredients.
    """
    recipe_set = set(i.lower() for i in recipe_ings)
    user_set   = set(i.lower() for i in user_ings_set)

    # Fuzzy matching — check if user ingredient is contained in recipe ingredient
    matched = set()
    missing = set()
    for r_ing in recipe_set:
        found = False
        for u_ing in user_set:
            if u_ing in r_ing or r_ing in u_ing:
                matched.add(r_ing)
                found = True
                break
        if not found:
            missing.add(r_ing)

    total    = len(recipe_set)
    coverage = (len(matched) / total * 100) if total > 0 else 0
    return round(coverage, 1), list(matched), list(missing)


def fetch_recipes_for_ingredients(user_ingredients, max_per_ingredient=5):
    """
    For each user ingredient, fetch matching meals from TheMealDB.
    Look up full details for each unique meal found.
    Cache everything in SQLite to avoid repeated API calls.
    Returns list of full meal dicts.
    """
    seen_ids = set()
    full_meals = []

    cached_df = load_cached_recipes()
    cached_ids = set(cached_df["meal_id"].tolist()) if not cached_df.empty else set()

    for ingredient in user_ingredients[:6]:  # limit to 6 ingredients to save API calls
        matches = api_filter_by_ingredient(ingredient.strip())
        for m in matches[:max_per_ingredient]:
            meal_id = m["idMeal"]
            if meal_id in seen_ids:
                continue
            seen_ids.add(meal_id)

            if meal_id in cached_ids:
                # Load from DB
                row = cached_df[cached_df["meal_id"] == meal_id].iloc[0]
                full_meals.append({
                    "source": "api",
                    "id":     meal_id,
                    "name":   row["name"],
                    "category": row["category"],
                    "area":   row["area"],
                    "instructions": row["instructions"],
                    "image":  row["image_url"],
                    "tags":   row["tags"],
                    "ingredients": row["ingredients"].split("|") if row["ingredients"] else [],
                    "youtube": row["youtube_url"],
                    "difficulty": "Medium",
                    "time_mins": None,
                })
            else:
                # Fetch from API and cache
                meal = api_lookup_by_id(meal_id)
                if meal:
                    save_api_recipe(meal)
                    full_meals.append({
                        "source": "api",
                        "id":     meal_id,
                        "name":   meal["strMeal"],
                        "category": meal.get("strCategory",""),
                        "area":   meal.get("strArea",""),
                        "instructions": meal.get("strInstructions",""),
                        "image":  meal.get("strMealThumb",""),
                        "tags":   meal.get("strTags",""),
                        "ingredients": parse_ingredients_from_meal(meal),
                        "youtube": meal.get("strYoutube",""),
                        "difficulty": "Medium",
                        "time_mins": None,
                    })

    return full_meals


def get_bonus_recipes():
    """Convert hardcoded bonus recipes to standard format."""
    result = []
    for r in BONUS_RECIPES:
        result.append({
            "source": "hardcoded",
            "id":     r["id"],
            "name":   r["name"],
            "category": r["category"],
            "area":   r["area"],
            "instructions": r["instructions"],
            "image":  r.get("image",""),
            "tags":   r.get("tags",""),
            "ingredients": r["ingredients"],
            "youtube": "",
            "difficulty": r.get("difficulty","Medium"),
            "time_mins": r.get("time_mins"),
        })
    return result


def match_recipes(user_ingredients, all_recipes, diet_filter="All",
                  cuisine_filter="All", min_coverage=20):
    """
    Main matching function.
    For each recipe, compute ingredient coverage score.
    Filter by diet and cuisine if specified.
    Return ranked list with coverage score and missing ingredients.
    """
    user_set = set(i.strip().lower() for i in user_ingredients if i.strip())
    results  = []

    for recipe in all_recipes:
        ings = recipe["ingredients"]
        if not ings:
            continue

        coverage, matched, missing = compute_coverage(user_set, ings)
        if coverage < min_coverage:
            continue

        # Diet filter (basic tag matching)
        tags = str(recipe.get("tags") or "").lower()
        cat  = str(recipe.get("category") or "").lower()
        if diet_filter == "Vegetarian" and "meat" in " ".join(ings):
            continue
        if diet_filter == "Vegetarian" and recipe["category"] in ["Beef","Chicken","Lamb","Pork","Seafood"]:
            continue
        if diet_filter == "Vegan" and not ("vegan" in tags):
            if recipe["category"] in ["Beef","Chicken","Lamb","Pork","Seafood","Dairy"]:
                continue

        # Cuisine filter
        if cuisine_filter != "All":
            if cuisine_filter.lower() not in str(recipe.get("area") or "").lower():
                continue

        results.append({
            **recipe,
            "coverage":  coverage,
            "matched":   matched,
            "missing":   missing,
            "missing_count": len(missing),
        })

    # Sort by coverage descending
    results.sort(key=lambda x: (-x["coverage"], x["missing_count"]))
    return results


# ─────────────────────────────────────────────
# ML: KNN RECIPE RECOMMENDATION
# Given user's saved favourites, find similar
# recipes using ingredient profile similarity.
# ─────────────────────────────────────────────

def build_ingredient_matrix(recipes, all_ingredients):
    """
    Build a binary ingredient matrix.
    Rows = recipes, Columns = ingredients (one-hot encoded).
    Used as input for KNN model.
    """
    mlb = MultiLabelBinarizer(classes=all_ingredients)
    ingredient_lists = [r["ingredients"] for r in recipes]
    return mlb.fit_transform(ingredient_lists), mlb


def recommend_by_knn(favourites_names, all_recipes, n=5):
    """
    KNN-based recipe recommendation.
    Finds recipes whose ingredient profiles are most
    similar to the user's saved favourites.

    Why KNN?
    - Ingredient lists are sparse binary vectors
    - KNN with cosine distance is ideal for sparse data
    - No training labels needed — purely similarity based

    Returns list of recommended recipes not already in favourites.
    """
    if not favourites_names or len(all_recipes) < 5:
        return []

    # Collect all unique ingredients across all recipes
    all_ings = sorted(set(
        ing for r in all_recipes for ing in r["ingredients"]
    ))
    if not all_ings:
        return []

    # Build ingredient matrix
    X, mlb = build_ingredient_matrix(all_recipes, all_ings)

    # Build query vector = average of favourite recipes' ingredient vectors
    fav_indices = [
        i for i, r in enumerate(all_recipes)
        if r["name"] in favourites_names
    ]
    if not fav_indices:
        return []

    query_vector = X[fav_indices].mean(axis=0).reshape(1, -1)

    # Fit KNN
    n_neighbors = min(n + len(fav_indices), len(all_recipes))
    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute"
    )
    knn.fit(X)
    distances, indices = knn.kneighbors(query_vector)

    # Return recipes not already in favourites
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        recipe = all_recipes[idx]
        if recipe["name"] not in favourites_names:
            recommendations.append({
                **recipe,
                "similarity": round((1 - dist) * 100, 1)
            })
        if len(recommendations) >= n:
            break

    return recommendations


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_coverage_bar(results):
    """Horizontal bar chart of ingredient coverage per recipe."""
    top = results[:15]
    names = [r["name"][:30] for r in top]
    coverages = [r["coverage"] for r in top]
    colors = ["#1D9E75" if c >= 80 else ("#f0c040" if c >= 50 else "#E8734A")
              for c in coverages]

    fig = go.Figure(go.Bar(
        x=coverages, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{c:.0f}%" for c in coverages],
        textposition="auto"
    ))
    fig.update_layout(
        title="Ingredient coverage — top matches",
        xaxis=dict(range=[0, 100], title="% ingredients you have"),
        height=max(300, len(top) * 28),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_category_donut(results):
    """Donut chart of recipe categories in results."""
    cats = {}
    for r in results:
        c = r.get("category") or "Other"
        cats[c] = cats.get(c, 0) + 1
    if not cats:
        return None
    fig = px.pie(
        values=list(cats.values()),
        names=list(cats.keys()),
        hole=0.45,
        title="Recipe categories",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
    return fig


def plot_missing_ingredients(results):
    """Bar chart of most commonly missing ingredients."""
    missing_count = {}
    for r in results[:20]:
        for ing in r.get("missing", []):
            missing_count[ing] = missing_count.get(ing, 0) + 1
    if not missing_count:
        return None
    top = sorted(missing_count.items(), key=lambda x: -x[1])[:10]
    fig = px.bar(
        x=[t[1] for t in top],
        y=[t[0] for t in top],
        orientation="h",
        labels={"x": "Recipes needing it", "y": ""},
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


def plot_cuisine_bar(results):
    """Bar chart of cuisine areas in results."""
    areas = {}
    for r in results:
        a = r.get("area") or "Other"
        areas[a] = areas.get(a, 0) + 1
    if not areas:
        return None
    sorted_areas = sorted(areas.items(), key=lambda x: -x[1])
    fig = px.bar(
        x=[a[0] for a in sorted_areas],
        y=[a[1] for a in sorted_areas],
        labels={"x": "Cuisine", "y": "Recipes"},
        title="Cuisines in your results",
        color=[a[1] for a in sorted_areas],
        color_continuous_scale="Teal"
    )
    fig.update_layout(
        height=280, showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10,r=10,t=40,b=10)
    )
    return fig


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def display_recipe_card(recipe, user_ings, show_save=True):
    """Render a full recipe card with ingredients, coverage and instructions."""
    coverage = recipe.get("coverage", 0)
    matched  = recipe.get("matched", [])
    missing  = recipe.get("missing", [])

    # Coverage color
    if coverage >= 80:
        cov_color = "🟢"
    elif coverage >= 50:
        cov_color = "🟡"
    else:
        cov_color = "🔴"

    with st.container():
        c1, c2 = st.columns([2, 1])

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
                f"{cov_color} **{coverage:.0f}% of ingredients covered** "
                f"({len(matched)}/{len(matched)+len(missing)} ingredients)"
            )

            # Matched ingredients
            if matched:
                matched_html = " ".join(
                    f'<span class="ingredient-tag">✅ {i}</span>'
                    for i in sorted(matched)
                )
                st.markdown(
                    f"**You have:** {matched_html}",
                    unsafe_allow_html=True
                )

            # Missing ingredients
            if missing:
                missing_html = " ".join(
                    f'<span class="missing-tag">❌ {i}</span>'
                    for i in sorted(missing)
                )
                st.markdown(
                    f"**Still need:** {missing_html}",
                    unsafe_allow_html=True
                )

        with c2:
            if recipe.get("image"):
                st.image(recipe["image"], use_container_width=True)
            if show_save:
                if st.button(f"❤️ Save", key=f"save_{recipe['id']}"):
                    save_favourite(recipe["id"], recipe["name"],
                                   recipe.get("source","api"))
                    st.success("Saved to favourites!")

        # Instructions expander
        with st.expander("📖 Full recipe instructions"):
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
        "Enter the ingredients you have at home and discover "
        "what you can cook. Powered by **TheMealDB API** + "
        "15 handcrafted Swiss & European recipes."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("🥕 Your Ingredients")
        st.markdown("Type the ingredients you have, one per line:")

        ingredients_text = st.text_area(
            "Ingredients",
            placeholder="chicken\nonion\ngarlic\ntomatoes\nolive oil",
            height=180,
            label_visibility="collapsed"
        )

        st.markdown("**Or add common ingredients quickly:**")
        quick_adds = ["chicken", "eggs", "pasta", "rice", "potatoes",
                      "onion", "garlic", "tomatoes", "cheese", "butter",
                      "milk", "flour", "beef", "salmon", "mushrooms"]
        quick_selected = []
        cols = st.columns(2)
        for i, ing in enumerate(quick_adds):
            if cols[i % 2].checkbox(ing, key=f"quick_{ing}"):
                quick_selected.append(ing)

        st.divider()
        st.header("🔧 Filters")

        diet_filter = st.selectbox(
            "Diet preference",
            ["All", "Vegetarian", "Vegan"]
        )
        cuisine_filter = st.selectbox(
            "Cuisine",
            ["All", "Italian", "French", "Spanish", "Swiss",
             "Asian", "American", "British", "Mexican",
             "Middle Eastern", "Mediterranean"]
        )
        min_coverage = st.slider(
            "Min ingredient coverage %",
            0, 100, 30, 5
        )
        include_bonus = st.checkbox(
            "Include Swiss/European bonus recipes", value=True)

        search_btn = st.button(
            "🔍 Find Recipes!", type="primary",
            use_container_width=True
        )

        st.divider()
        st.markdown("**💡 Tips:**")
        st.caption(
            "- Set coverage to 0% to see ALL possible recipes\n"
            "- Set to 80%+ for recipes you can make right now\n"
            "- Missing ingredients are shown in red"
        )

    # ── Parse ingredients ──
    typed = [i.strip().lower() for i in ingredients_text.split("\n")
             if i.strip()] if ingredients_text else []
    user_ingredients = list(set(typed + quick_selected))

    if not user_ingredients and not search_btn:
        st.info("👈 Enter your ingredients in the sidebar and click **Find Recipes!**")

        # Show random recipe while waiting
        st.subheader("🎲 Recipe of the day")
        with st.spinner("Loading..."):
            random_meal = api_get_random()
        if random_meal:
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.markdown(f"### {random_meal['strMeal']}")
                st.markdown(
                    f"🌍 {random_meal.get('strArea','')} | "
                    f"🍽️ {random_meal.get('strCategory','')}"
                )
                with st.expander("📖 Instructions"):
                    st.markdown(random_meal.get("strInstructions",""))
            with rc2:
                if random_meal.get("strMealThumb"):
                    st.image(random_meal["strMealThumb"],
                             use_container_width=True)
        return

    if not user_ingredients:
        st.warning("Please enter at least one ingredient.")
        return

    # ── Tabs ──
    tab_results, tab_analytics, tab_ml, tab_favs, tab_rate, tab_log = st.tabs([
        "🍽️ Recipes",
        "📊 Analytics",
        "🤖 ML Recommendations",
        "❤️ Favourites",
        "⭐ Rate Recipes",
        "🕒 Search History"
    ])

    # ── Fetch and match ──
    with st.spinner("Searching recipes..."):
        api_recipes = fetch_recipes_for_ingredients(user_ingredients)
        bonus = get_bonus_recipes() if include_bonus else []
        all_recipes = api_recipes + bonus

        matched_recipes = match_recipes(
            user_ingredients, all_recipes,
            diet_filter, cuisine_filter, min_coverage
        )

    # Log search
    top_name = matched_recipes[0]["name"] if matched_recipes else "None"
    log_search(user_ingredients, diet_filter, cuisine_filter,
               len(matched_recipes), top_name)

    # ── TAB 1: RESULTS ──
    with tab_results:
        st.subheader(
            f"🍽️ Found {len(matched_recipes)} recipes "
            f"for: {', '.join(user_ingredients[:5])}"
            + ("..." if len(user_ingredients) > 5 else "")
        )

        if not matched_recipes:
            st.warning(
                "No recipes found. Try lowering the coverage % "
                "or adding more ingredients."
            )
        else:
            # Quick stats
            qs1, qs2, qs3, qs4 = st.columns(4)
            can_make_now = [r for r in matched_recipes if r["coverage"] >= 80]
            qs1.metric("Total matches",   len(matched_recipes))
            qs2.metric("Can make NOW 🟢", len(can_make_now))
            qs3.metric("Best coverage",
                       f"{matched_recipes[0]['coverage']:.0f}%")
            qs4.metric("Your ingredients", len(user_ingredients))

            st.divider()

            # Sort options
            sort_by = st.radio(
                "Sort by:",
                ["Coverage (best first)", "Fewest missing ingredients",
                 "Alphabetical"],
                horizontal=True
            )
            if sort_by == "Fewest missing ingredients":
                matched_recipes = sorted(
                    matched_recipes, key=lambda x: x["missing_count"])
            elif sort_by == "Alphabetical":
                matched_recipes = sorted(
                    matched_recipes, key=lambda x: x["name"])

            # Show results
            for recipe in matched_recipes:
                display_recipe_card(recipe, user_ingredients)

    # ── TAB 2: ANALYTICS ──
    with tab_analytics:
        st.subheader("📊 Recipe Analytics")

        if not matched_recipes:
            st.info("Run a search first to see analytics.")
        else:
            ac1, ac2 = st.columns(2)
            with ac1:
                st.plotly_chart(
                    plot_coverage_bar(matched_recipes),
                    use_container_width=True, key="cov_bar"
                )
            with ac2:
                fig_cat = plot_category_donut(matched_recipes)
                if fig_cat:
                    st.plotly_chart(fig_cat, use_container_width=True,
                                    key="cat_donut")

            ac3, ac4 = st.columns(2)
            with ac3:
                fig_miss = plot_missing_ingredients(matched_recipes)
                if fig_miss:
                    st.plotly_chart(fig_miss, use_container_width=True,
                                    key="miss_bar")
            with ac4:
                fig_cuis = plot_cuisine_bar(matched_recipes)
                if fig_cuis:
                    st.plotly_chart(fig_cuis, use_container_width=True,
                                    key="cuis_bar")

            st.divider()
            st.subheader("🛒 Shopping list")
            st.markdown(
                "If you want to cook **all** matched recipes, "
                "you still need:"
            )
            all_missing = set()
            for r in matched_recipes:
                all_missing.update(r.get("missing", []))
            user_set = set(i.lower() for i in user_ingredients)
            truly_missing = [
                i for i in sorted(all_missing)
                if not any(u in i or i in u for u in user_set)
            ]
            if truly_missing:
                cols = st.columns(3)
                for i, ing in enumerate(truly_missing):
                    cols[i % 3].markdown(f"- {ing}")
            else:
                st.success("You have everything you need! 🎉")

    # ── TAB 3: ML RECOMMENDATIONS ──
    with tab_ml:
        st.subheader("🤖 KNN Recipe Recommendations")
        st.markdown(
            "Based on your **saved favourites**, our K-Nearest Neighbors "
            "model finds recipes with similar ingredient profiles — "
            "recipes you'd likely enjoy but haven't tried yet."
        )

        df_favs = load_favourites()
        if df_favs.empty:
            st.info(
                "Save some favourite recipes first (click ❤️ Save on any recipe) "
                "and come back here for personalised recommendations!"
            )
        else:
            fav_names = df_favs["recipe_name"].tolist()
            st.markdown(
                f"Based on your {len(fav_names)} saved favourites: "
                f"**{', '.join(fav_names[:5])}**"
                + ("..." if len(fav_names) > 5 else "")
            )

            with st.spinner("Running KNN model..."):
                recommendations = recommend_by_knn(
                    fav_names, all_recipes, n=6)

            if not recommendations:
                st.info("Not enough recipe data yet for recommendations.")
            else:
                st.success(
                    f"Found {len(recommendations)} recipes similar "
                    f"to your favourites!"
                )
                for rec in recommendations:
                    st.markdown(
                        f"**{rec['name']}** — "
                        f"{rec.get('area','')} {rec.get('category','')} | "
                        f"Similarity: {rec.get('similarity',0):.0f}%"
                    )
                    display_recipe_card(rec, user_ingredients)

            st.markdown("""
            ---
            **💡 How KNN works here:**
            Each recipe is represented as a binary vector where each position
            corresponds to an ingredient (1 = recipe contains it, 0 = doesn't).
            KNN finds recipes whose ingredient vectors are most similar
            (smallest cosine distance) to the average vector of your favourites.
            This is genuine unsupervised similarity search — no manual rules.
            """)

    # ── TAB 4: FAVOURITES ──
    with tab_favs:
        st.subheader("❤️ Your Saved Recipes")
        df_favs = load_favourites()
        if df_favs.empty:
            st.info("No saved recipes yet. Click ❤️ Save on any recipe!")
        else:
            st.dataframe(
                df_favs[["recipe_name","source","saved_at"]].rename(columns={
                    "recipe_name":"Recipe",
                    "source":"Source",
                    "saved_at":"Saved at"
                }),
                use_container_width=True, hide_index=True
            )
            if st.button("🗑️ Clear favourites"):
                conn = sqlite3.connect("cookbook.db")
                conn.execute("DELETE FROM favourites")
                conn.commit()
                conn.close()
                st.rerun()

    # ── TAB 5: RATE RECIPES ──
    with tab_rate:
        st.subheader("⭐ Rate Recipes You've Tried")

        if not matched_recipes:
            st.info("Search for recipes first, then rate the ones you've tried.")
        else:
            recipe_to_rate = st.selectbox(
                "Which recipe did you try?",
                [r["name"] for r in matched_recipes]
            )
            rating = st.slider("Your rating", 1, 5, 3)
            stars = "⭐" * rating
            st.markdown(f"Rating: {stars}")
            notes = st.text_input("Notes (optional)",
                                  placeholder="Delicious! I added extra garlic...")
            if st.button("✅ Submit rating"):
                recipe_id = next(
                    (r["id"] for r in matched_recipes
                     if r["name"] == recipe_to_rate), recipe_to_rate
                )
                save_rating(recipe_id, recipe_to_rate, rating, notes)
                st.success(f"Rated {recipe_to_rate}: {stars}")

            st.divider()
            st.subheader("Your ratings")
            df_ratings = load_ratings()
            if df_ratings.empty:
                st.info("No ratings yet.")
            else:
                df_ratings["stars"] = df_ratings["rating"].apply(
                    lambda r: "⭐" * int(r))
                st.dataframe(
                    df_ratings[["recipe_name","stars","notes","rated_at"]].rename(
                        columns={"recipe_name":"Recipe","stars":"Rating",
                                 "notes":"Notes","rated_at":"Date"}),
                    use_container_width=True, hide_index=True
                )

    # ── TAB 6: SEARCH LOG ──
    with tab_log:
        st.subheader("🕒 Search History")
        df_log = load_search_log()
        if df_log.empty:
            st.info("No searches yet.")
        else:
            st.dataframe(df_log, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear history"):
            conn = sqlite3.connect("cookbook.db")
            conn.execute("DELETE FROM search_log")
            conn.commit()
            conn.close()
            st.rerun()


if __name__ == "__main__":
    main()
