# Podcast AdBlocker – IDATT2502

Created by:
- Elias Trana, elias.t.trana@ntnu.no
- Erik Nordsæther, eriktn@stud.ntnu.no
- Sander Rom Skofsrud, sander.r.skofsrud@ntnu.no
- Vegard Johnsen, vegard.johnsen@ntnu.no

### Application
Podcast AdBlocker is a simple Python script that removes ads from podcasts.
It works by downloading the podcast, removing the ads, and then re-uploading the podcast without ads.
The application utilizes Flask as the backend and Next.js as the frontend. 
### Pipeline
It also includes a complex pipeline
for testing different Whisper and OpenAI models to determine the best model for removing ads. This pipeline utilizes given files as testdata,
and constructs a detailed HTML page with graphs, statistics and interactive results of accuracy and speed. 

## Installation

To install Podcast AdBlocker, start by cloning the repository:

```bash
git clone https://github.com/SanderSkofsrud/podcast.git
```

Then, decide what you want to do:

- If you want to use the program, read the README.md file in the `Application` directory.
- If you want to test the pipeline, read the README.md file in the `Pipeline` directory.

### Important:

If you want to run the program, you will need an OPENAI API key. If you do not have one yourself, please contact one of the developers to borrow one for testing purposes.



