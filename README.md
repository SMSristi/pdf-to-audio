# 🎧 PDF to Audio Converter

A simple web app built with **Python** and **Streamlit** that converts PDF files into audio using **text-to-speech (TTS)**.

---

## 📌 Features

- Upload a small PDF file
- Extracts text from the PDF
- Converts it to audio using Google Text-to-Speech (gTTS)
- Streams the generated audio in the browser

---

## ⚠️ Notes & Limitations

- Best for **short PDFs** (less than ~10 pages)
- May fail or slow down on long PDFs or scanned documents
- Uses free gTTS API — has request limits

---


## 🧱 Project Structure

```
pdf-to-audio/
├── app.py ← Streamlit web app
├── pdf_to_audio.py ← Original CLI script using pyttsx3
├── requirements.txt
└── README.md
```

---

## 🖥️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/SMSristi/pdf-to-audio.git
cd pdf-to-audio
Install dependencies
```

```bash
pip install -r requirements.txt
Run the app
```

```bash
streamlit run app.py
```

💻 CLI Version (Offline)
There’s also a basic command-line version in pdf_to_audio.py which uses the pyttsx3 TTS engine (offline, no internet required).

Run it with:

```bash
python pdf_to_audio.py
```
Make sure to adjust the file path inside the script.

If you found this project useful, feel free to ⭐ star the repo or share it!

## 📜 License

This project is open-source under the MIT License.

