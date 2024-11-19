# How to run the pipeline

This is a pipeline used to test the different Whisper and OpenAI api models.

To run the pipline, do the following commands:

``````
docker-compose build
docker-compose up podcast_processor
``````

The application should initialize and run with only this command.

If you want to run any of the legacy models, used for testing specific features, you can run the following commands:

``````
docker-compose up legacy_llm
``````
or 
``````
docker-compose up legacy_transcription
``````

Note, that you need both an OPENAI_API_KEY and an LLAMA_API_KEY to run the legacy_llm. If you do not have one yourself, please contact one of the developers to borrow them for testing purposes.