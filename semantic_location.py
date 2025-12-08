import re
import pandas as pd

df = pd.read_csv("refined_data.csv")

def categorize_address(addr):
    if pd.isna(addr):
        return "unknown"
    
    a = addr.lower()

    # HOME / APARTMENTS / RESIDENCE
    if any(x in a for x in ["apartment", "apt", "residence", "home", "house", "complex"]):
        return "home"

    # UNIVERSITY / CAMPUS / ACADEMIC BUILDINGS
    if any(x in a for x in ["university", "tamu", "college", "campus", "engineering", "building", "hall"]):
        return "university"

    # CAFES / COFFEE SHOPS
    if any(x in a for x in ["starbucks", "coffee", "cafe"]):
        return "cafe"

    # RESTAURANTS / FOOD PLACES
    if any(x in a for x in ["restaurant", "food", "grill", "pizza", "burger", "chipotle", "chick-fil-a"]):
        return "restaurant"

    # SHOPS / STORES / GROCERIES
    if any(x in a for x in ["store", "market", "heb", "walmart", "kroger", "mall", "shopping"]):
        return "store"

    # PARKING AREAS
    if "parking" in a:
        return "parking"

    # GYMS / FITNESS / SPORTS
    if any(x in a for x in ["gym", "fitness", "recreation", "rec center"]):
        return "gym"

    # HOSPITALS / CLINICS
    if any(x in a for x in ["hospital", "clinic", "medical"]):
        return "medical"

    # BARS / NIGHTLIFE
    if any(x in a for x in ["bar", "pub", "night", "club"]):
        return "nightlife"

    # WORKPLACES (OFFICES, COMPANIES)
    if any(x in a for x in ["office", "company", "corporate"]):
        return "office"

    # SCHOOLS (K–12)
    if any(x in a for x in ["high school", "middle school", "elementary"]):
        return "school"

    return "other"

df["semantic_location"] = df["Named Location Category"].apply(categorize_address)
df["semantic_location"].value_counts()