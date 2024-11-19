# How to run the application

This is the web-application for the project. It lets the user select a file from their computer and have its ads be removed. 

In the Application directory, run this command:

``````
docker-compose -f production.yml up --build -d
``````

The application should initialize and run with only this command.

### Important: 
When the docker container is starting up, the frontend will be availible first, followed by the backend. There is a slight delay before the flask backend is operational. It is ready for use when you see this inside the docker console:

``````
2024-11-18 12:28:40 flask_backend    |  * Running on all addresses (0.0.0.0)
2024-11-18 12:28:40 flask_backend    |  * Running on http://127.0.0.1:5001
2024-11-18 12:28:40 flask_backend    |  * Running on http://172.23.0.3:5001
2024-11-18 12:28:40 flask_backend    | INFO:werkzeug:Press CTRL+C to quit
``````