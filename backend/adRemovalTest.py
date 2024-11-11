import openai
from dotenv import load_dotenv
from difflib import SequenceMatcher, get_close_matches
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

load_dotenv()
openai.api_key = "key"


def read_transcription(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


podcasts = [
    {
        'name': 'Kort podcast (30min)',
        'with_ads': 'podTranscripWithAds.txt',
        'without_ads': 'podTranscript.txt'
    },
    {
        'name': 'Medium podcast (1t)',
        'with_ads': 'pod3.txt',
        'without_ads': 'pod3NoAds.txt'
    },
    {
        'name': 'Lang podcast (2.2t)',
        'with_ads': 'pod2.txt',
        'without_ads': 'pod2NoAds.txt'
    }
]

def test_model_performance(model_name, text):
    prompt = (
        "The following is a podcast transcription that contains some advertisements. "
        "Please identify and list the advertisements in this text.\n\n"
        f"{text}\n\n"
        "Provide a copy of the ads in the transcript."
    )

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that helps identify and remove advertisements in podcast transcriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.5,
            )
            ads = response.choices[0].message["content"].strip()
            return ads
        except openai.error.RateLimitError as e:
            wait_time = re.search(r"Please try again in (\d+\.\d+)s", str(e))
            if wait_time:
                wait_seconds = float(wait_time.group(1))
                print(f"Rate limit nådd for {model_name}. Venter i {wait_seconds:.2f} sekunder...")
                time.sleep(wait_seconds)
            else:
                print(f"Rate limit nådd for {model_name}. Venter i 60 sekunder...")
                time.sleep(60)
        except openai.error.Timeout:
            print(f"Timeout oppstod for {model_name}. Venter i 10 sekunder før nytt forsøk...")
            time.sleep(10)
        except Exception as e:
            print(f"En feil oppstod for {model_name}: {e}")
            return ""

def remove_ads_from_transcription(transcription, ads):
    for ad in ads.split("\n"):
        ad = ad.strip()
        if ad:
            close_matches = get_close_matches(ad, transcription.split("\n"), cutoff=0.5)
            for match in close_matches:
                transcription = transcription.replace(match, "")
    return transcription.strip()

def compare_transcriptions(modified_text, original_text):
    matcher = SequenceMatcher(None, modified_text, original_text)
    similarity_ratio = matcher.ratio()
    return similarity_ratio

def split_text(text, max_tokens=2000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sentence + '. '
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + '. '
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

models = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o-mini",
    "gpt-4o",
]

num_runs = 1

results = []

for podcast in podcasts:
    print(f"\nBehandler {podcast['name']}...\n")

    transcription_with_ads = read_transcription(podcast['with_ads'])
    transcription_without_ads = read_transcription(podcast['without_ads'])


    transcription_chunks = split_text(transcription_with_ads, max_tokens=1500)

    for model in models:
        print(f"Tester modell {model} på {podcast['name']} over {num_runs} kjøringer...\n")

        for run in range(num_runs):
            print(f"Kjøring {run + 1}/{num_runs}...")
            try:
                start_time = time.time()

                identified_ads_list = []

                for chunk in transcription_chunks:
                    ads = test_model_performance(model, chunk)
                    identified_ads_list.append(ads)
                    time.sleep(1)

                identified_ads = '\n'.join(identified_ads_list)

                modified_transcription = remove_ads_from_transcription(
                    transcription_with_ads, identified_ads
                )

                end_time = time.time()
                elapsed_time = end_time - start_time

                similarity = compare_transcriptions(
                    modified_transcription, transcription_without_ads
                )


                results.append({
                    'Podcast': podcast['name'],
                    'Modell': model,
                    'Kjøring': run + 1,
                    'Likhet': similarity,
                    'Tid (sekunder)': elapsed_time
                })

                print(
                    f"Likhet: {similarity:.2f}, Tid: {elapsed_time:.2f} sekunder\n"
                )

            except Exception as e:
                print(f"En feil oppstod med modell {model} i kjøring {run + 1}: {e}\n")
                continue


results_df = pd.DataFrame(results)


results_df.to_csv('modell_podcast_resultater.csv', index=False)

# plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=results_df, x='Kjøring', y='Likhet', hue='Modell', style='Podcast', markers=True, dashes=False)
plt.title('Likhet over flere kjøringer per modell og podcast')
plt.xlabel('Kjøring')
plt.ylabel('Likhet')
plt.ylim(0, 1)
plt.legend(title='Modell og Podcast', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(data=results_df, x='Kjøring', y='Tid (sekunder)', hue='Modell', style='Podcast', markers=True, dashes=False)
plt.title('Tid brukt over flere kjøringer per modell og podcast')
plt.xlabel('Kjøring')
plt.ylabel('Tid (sekunder)')
plt.legend(title='Modell og Podcast', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Modell', y='Likhet', hue='Podcast', errorbar='sd', capsize=.2)
plt.title('Gjennomsnittlig likhet per modell og podcast med standardavvik')
plt.ylabel('Gjennomsnittlig likhet')
plt.ylim(0, 1)
plt.legend(title='Podcast', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Modell', y='Tid (sekunder)', hue='Podcast', errorbar='sd', capsize=.2)
plt.title('Gjennomsnittlig tid per modell og podcast med standardavvik')
plt.ylabel('Gjennomsnittlig tid (sekunder)')
plt.legend(title='Podcast', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Tid (sekunder)', y='Likhet', hue='Modell', style='Podcast', s=100)
plt.title('Forhold mellom tid og likhet per modell og podcast')
plt.xlabel('Tid (sekunder)')
plt.ylabel('Likhet')
plt.legend(title='Modell og Podcast', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()