import pandas as pd

# Let's debug the genre parsing issue
print("🔍 Debugging genre processing...")

# Load data
df = pd.read_csv('data/raw/movies_popular.csv')

print(f"\n📊 Sample genre data:")
for i in range(5):
    genre_value = df.iloc[i]['genres']
    print(f"Movie {i+1}: {df.iloc[i]['title']}")
    print(f"  Raw genre value: {repr(genre_value)}")
    print(f"  Type: {type(genre_value)}")
    
    # Try to process it
    try:
        if isinstance(genre_value, str):
            parsed = eval(genre_value)
            print(f"  Parsed as: {parsed}")
            print(f"  Parsed type: {type(parsed)}")
        else:
            print(f"  Already parsed: {genre_value}")
    except Exception as e:
        print(f"  Error parsing: {e}")
    print()

print(f"📋 All unique genre values (first 10):")
unique_genres = df['genres'].unique()
for i, genre in enumerate(unique_genres[:10]):
    print(f"{i+1}. {repr(genre)}")

# Test our genre extraction logic
print(f"\n🧪 Testing genre extraction:")
from collections import defaultdict

genre_stats = defaultdict(list)

for idx, movie in df.iterrows():
    try:
        genres = movie['genres']
        print(f"Processing: {movie['title']}")
        print(f"  Genres: {repr(genres)}")
        
        # Handle different formats
        if isinstance(genres, str):
            if genres.startswith('['):
                genres_list = eval(genres)
            else:
                genres_list = [genres]  # Single genre as string
        elif isinstance(genres, list):
            genres_list = genres
        else:
            genres_list = []
            
        print(f"  Processed as: {genres_list}")
        
        for genre in genres_list:
            genre_stats[genre].append(movie['title'])
            
    except Exception as e:
        print(f"  Error: {e}")
    
    if idx >= 3:  # Just test first few
        break

print(f"\n📊 Extracted genres:")
for genre, movies in genre_stats.items():
    print(f"  {genre}: {len(movies)} movies")