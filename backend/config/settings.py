from dotenv import load_dotenv
import os

load_dotenv()

def get_speech_key():
    return os.getenv("SPEECH_KEY")

def get_speech_region():
    return os.getenv("SPEECH_REGION")

def get_openai_key():
    return os.getenv("AZURE_OPENAI_API_KEY")

def get_openai_endpoint():
    return os.getenv("AZURE_OPENAI_ENDPOINT")

def get_openai_version():
    return os.getenv("AZURE_OPENAI_VERSION")

def get_openai_name():
    return os.getenv("AZURE_OPENAI_NAME")

from openai import AzureOpenAI

# Sett opp klienten
client = AzureOpenAI(
    api_key=get_openai_key(),  
    api_version=get_openai_version(),
    azure_endpoint=get_openai_endpoint(),  
    azure_deployment=get_openai_name()  
)

"""


# Example request
completion = client.chat.completions.create(
    model=get_openai_name(),  
    messages=[
        {"role": "system", "content": "Du er en tekstklassifikator som avgjør om en tekst er en annonse eller ikke."},
        {"role": "user", "content": "Skriv en kort annonse for et sommertilbud."}
    ],
    max_tokens=50
)

print(completion.choices[0].message.content)


"""









