# Final-Year-Project-Term-2

1. Put the audio files in a folder into the temp folder
2. execute split.py to obtain, the preprocessed audio files will be inside temp_cutted folder
3. Move the processed training folder to 'data' folder.
4. Execute preprocessing.py for obtaining the np array file in the root
5. Execut training.py to train the model, the model will be saved in root directory.
6. Move the evaluation audio file folder into test folder.
7. Execute evalution.py to obtain a spreadsheet of the result and a result.txt file for application processing.
8. Execute trime_by_time.py to obtain a splitted segments with name of the starting timestamp.Speech segment will saved in for_transcript folder. Music will saved in music_extracted folder.
9. Execute google_speech.py to obtain transcript of each segment.
10. Execute transcript_to _subtitle.py to obtain subtitle file for the evaluation audio.

File structure

data<br />
├──musan<br />
├──────speech<br />
├────────speech_folders<br />
├───────────speech files in each folders<br />
├──────music<br />
├────────music_folders<br />
├───────────music files in each folders<br />
temp<br />
├──unprocessed audio folder<br />
├──────unprocessed audio or video<br />
temp_cutted<br />
├──processed audio folder<br />
├──────processed audio<br />
test<br />
├──evaluation folder<br />
├──────evalation audio file<br />
    
