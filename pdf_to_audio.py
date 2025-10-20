import PyPDF2
import pyttsx3
import os

# Initialize speaker
speaker = pyttsx3.init()

# Optional: set voice or speed
# voices = speaker.getProperty('voices')
# speaker.setProperty('voice', voices[1].id)
# speaker.setProperty('rate', 150)

# Read and process the PDF inside the 'with' block
full_text = ""
with open('sample.pdf', 'rb') as file:
    pdf_Reader = PyPDF2.PdfReader(file)

    for page_num in range(len(pdf_Reader.pages)):
        text = pdf_Reader.pages[page_num].extract_text()
        if text:
            full_text += text

# Save to MP3 after reading the PDF
speaker.save_to_file(full_text, 'audio.mp3')
speaker.runAndWait()

# Optionally play the file
os.startfile('audio.mp3')

print("✅ Audiobook saved and played successfully!")



#No Saving

# import PyPDF2
# import pyttsx3

# # Read the pdf by specifying the path in your computer
# pdf_Reader = PyPDF2.PdfReader(open('Submission.pdf', 'rb'))

# # Get the handle to speaker
# speaker = pyttsx3.init()

# # Split the pages and read one by one
# for page_num in range(len(pdf_Reader.pages)):
#     text = pdf_Reader.pages[page_num].extract_text()
#     speaker.say(text)  
#     speaker.runAndWait()

# # Stop the speaker after completion
# speaker.stop()

# # Save the audiobook at specified path
# # engine = pyttsx3.init()
# engine.save_to_file(text, 'audio.mp3')
# engine.runAndWait()