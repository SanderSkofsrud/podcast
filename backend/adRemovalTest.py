import openai
from dotenv import load_dotenv
from difflib import SequenceMatcher, get_close_matches
import time

load_dotenv()
openai.api_key = "key"


def read_transcription(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


transcription_with_ads = read_transcription("podTranscripWithAds.txt")
transcription_without_ads = read_transcription("podTranscript.txt")

# idntify ads
def test_model_performance(model_name, text):
    prompt = (
        "The following is a podcast transcription that contains some advertisements. "
        "Please identify and list the advertisements in this text.\n\n"
        f"{text}\n\n"
        "Provide a copy of the ads in the transcript."
    )

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an assistant that helps identify and remove advertisements in podcast transcriptions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5
    )

    ads = response.choices[0].message['content'].strip()
    return ads

# remove ads
def remove_ads_from_transcription(transcription, ads):
    for ad in ads.split("\n"):
        ad = ad.strip()
        if ad:

            close_matches = get_close_matches(ad, transcription.split('\n'), cutoff=0.5)
            for match in close_matches:
                transcription = transcription.replace(match, "")
    return transcription.strip()


# compare transcriptions
def compare_transcriptions(modified_text, original_text):
    similarity_ratio = SequenceMatcher(None, modified_text, original_text).ratio()
    return similarity_ratio


models = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o-mini",
    "gpt-4o"
]

# test performance
for model in models:
    print(f"Testing {model}...")

    # Start time measurement
    start_time = time.time()

    # Identify ads
    identified_ads = test_model_performance(model, transcription_with_ads)
    print(f"Advertisements identified by {model}:\n{identified_ads}\n")

    # Remove ads from transcription
    modified_transcription = remove_ads_from_transcription(transcription_with_ads, identified_ads)

    # End time measurement
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compare modified transcription with original (ad-free) transcription
    similarity = compare_transcriptions(modified_transcription, transcription_without_ads)
    print(f"Similarity between modified and original (ad-free) transcription: {similarity:.2f}")
    print(f"Time taken by {model} to remove ads: {elapsed_time:.2f} seconds\n")
