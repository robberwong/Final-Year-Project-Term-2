import argparse
import io
import os
from pydub import AudioSegment
from pydub import silence
from pydub.utils import make_chunks
import argparse
folder='for_transcript/'
f = open("recognition.txt", "w",encoding='utf8')
def transcribe_onprem(local_file_path, index):

    from google.cloud import speech

    client = speech.SpeechClient.from_service_account_json('service_account.json')

    # The language of the supplied audio
    language_code = "en-us"

    # Sample rate in Hertz of the audio data sent
    sample_rate = 16000



    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate,
    }
    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}
    response = client.recognize(request={"config": config, "audio": audio})
    f = open("recognition.txt", "a",encoding='utf8')
    f.writelines(str(index)+':')
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(f"Transcript: {alternative.transcript}")
        f.writelines(f"{alternative.transcript}"+' ')
    f.writelines('\n')
    f.close()



# [END speech_transcribe_async_word_time_offsets_gcs]


if __name__ == "__main__":
    '''Splitting segment to 10 seconds each'''
    for file in os.listdir(folder):
        print(file)
        audio = AudioSegment.from_file(folder+'/'+file )
        audio = audio.set_frame_rate(16000)
        start_trim = silence.detect_leading_silence(audio)
        end_trim = silence.detect_leading_silence(audio.reverse())
        duration = len(audio)
        audio = audio[start_trim:duration-end_trim]
        audio=audio.set_channels(1)
        chunk_length_ms = 10000 # pydub calculates in millisec
        #chunks = split_on_silence(myaudio,min_silence_len=500,silence_thresh=-16,keep_silence = 100)
        chunks = make_chunks(audio, chunk_length_ms) 
        
        '''output all the audio'''
        for i, chunk in enumerate(chunks):
            chunk_name ="{num:010d}".format(num=int(file[:-4])+16000*chunk_length_ms//1000*i)
            #print ("exporting "+chunk_name)
            if not os.path.exists('transcript_cutted/'):
                os.makedirs('transcript_cutted/')
            chunk.export('transcript_cutted/'+chunk_name+'.wav', format="wav",)
        
    '''obtain the transcript of each files'''    
    for file in os.listdir('transcript_cutted/'): 
        print(file)     
        transcribe_onprem(local_file_path='transcript_cutted/'+file, index=int(file[:-4]))