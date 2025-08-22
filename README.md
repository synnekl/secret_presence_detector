# "Secret" detector and recorder

This is the evil exam project of group 5. The project involves developing a key word detection model that responds to the word "secret". The aim is to deploy the model to an Arduino so that it kan listen for the word "secret" in conversation and then start to record until recieving an end-recording signal - a discreet wave. The arduino will use the microphone and motion sensor. The model is developed and trained on computer before being compressed to fit on the Arduino. 
Ideally, both audio detection and keyword recognition should run locally on the device, while the recorded audio is sent to a laptop for further processing, such as transcription or comparison between the two devices.

The project includes:
- Each of the team members employ dummy models to the Arduino to test the different sensor groups (environment, sound, motion and gesture)
- Creating a novel database of 1s 1600hz audio clips of the word "secret"
- Training the tflite_model_maker audio_classifier with the data, a lightweight CNN-based audio classifier with new dataset and logging performance
- Documenting the process on GitHub

Files in this GitHub include: 
- MLOps_DTU_secrets_WAV.zip: a zip-folder containing .wav files of 1s 1600 hz audioclips of the word "secret" by different people
- speech_recognition.ipynb: Code for training key word recognition model from the new dataset and two public datasets and compressiong to a 8bit tflite model
- 
