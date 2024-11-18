# How to run the pipeline

This is a pipeline used to test the different Whisper and OpenAI api models.

To run the pipline, do the following commands:

``````
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