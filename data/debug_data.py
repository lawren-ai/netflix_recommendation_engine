import pandas as pd

# Let's investigate our data
print("🔍 Debugging our movie data...")

# Load and examine the CSV
df = pd.read_csv('data/raw/movies_popular.csv')

print(f"\n📊 Dataset Info:")
print(f"  • Total movies: {len(df)}")
print(f"  • Columns: {list(df.columns)}")

print(f"\n📋 Sample movie titles:")
for i, title in enumerate(df['title'].head(15)):
    print(f"  {i+1}. {title}")

print(f"\n🔍 Data types:")
print(df.dtypes)

print(f"\n❓ Looking for our test movies:")
test_movies = ["Avengers", "Toy Story", "Titanic", "Avatar", "Spider", "Batman"]

for movie in test_movies:
    matches = df[df['title'].str.contains(movie, case=False, na=False)]
    if not matches.empty:
        print(f"✅ Found matches for '{movie}':")
        for title in matches['title'].head(3):
            print(f"   • {title}")
    else:
        print(f"❌ No matches for '{movie}'")

print(f"\n📊 Sample of genres data:")
print(df['genres'].head(5))