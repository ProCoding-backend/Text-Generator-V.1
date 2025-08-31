# dataset_file.py
# Generates a richer dataset with multiple categories
import random

def generate_dataset(filename="data.txt", num_lines=800):
    # Define multiple categories of "keys"
    story_lines = [
        "Once upon a time, a hero walked into the unknown.",
        "The village celebrated under the banyan tree every evening.",
        "A traveler shared wisdom with those willing to listen.",
        "Children laughed as they chased butterflies in the meadow.",
        "An ancient book whispered stories of forgotten kings."
    ]

    wisdom_lines = [
        "Knowledge grows when it is shared.",
        "Imagination is stronger than fear.",
        "Unity brings strength, division brings weakness.",
        "Kindness is the greatest magic.",
        "Learning never ends, it evolves."
    ]

    science_lines = [
        "The Earth orbits the Sun once every year.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Stars are giant spheres of burning plasma.",
        "Gravity pulls everything with mass toward each other.",
        "Electricity flows through conductors like copper."
    ]

    fantasy_lines = [
        "Dragons soared above the mountains breathing fire.",
        "The wizard raised his staff and cast a spell.",
        "Hidden treasures sparkled deep within the cave.",
        "The enchanted forest glowed under the moonlight.",
        "A portal opened to another magical realm."
    ]

    conversation_lines = [
        "A: Hello, how are you today?",
        "B: I'm fine, thank you. And you?",
        "A: Did you hear the story of the old book?",
        "B: Yes, it was amazing. I love stories!",
        "A: Let's write our own tale together."
    ]

    # Merge all categories into one big pool
    categories = [story_lines, wisdom_lines, science_lines, fantasy_lines, conversation_lines]

    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            # Pick a random category and then a random line from it
            category = random.choice(categories)
            line = random.choice(category)
            f.write(line + "\n")

    print(f"✅ Generated dataset with {num_lines} lines at {filename}")

if __name__ == "__main__":
    generate_dataset("data.txt", num_lines=800)
