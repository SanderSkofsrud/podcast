from dotenv import load_dotenv
import os

load_dotenv()


def get_openai_key():
    return os.getenv("AZURE_OPENAI_API_KEY")

def get_openai_endpoint():
    return os.getenv("AZURE_OPENAI_ENDPOINT")

def get_openai_version():
    return os.getenv("AZURE_OPENAI_VERSION")

def get_openai_name():
    return os.getenv("AZURE_OPENAI_NAME")


from openai import OpenAI

# Sett opp klienten
client = OpenAI()










