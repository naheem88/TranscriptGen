# Audio Transcription Tool

A Python script that transcribes video files using Google's Gemini 2.0 Flash API. Handles long recordings by splitting them into chunks and preserving context between segments.

## What it does

- Extracts audio from video files 
- Splits long audio into manageable chunks with overlap
- Transcribes each chunk using Gemini 2.0 Flash
- Combines transcripts and removes duplicate content
- Supports parallel processing for faster results

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install ffmpeg (required for audio processing):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/

3. Create a `.env` file with your Google API key:
```
GENAI_API_KEY=your_api_key_here
```

## Usage

Basic usage:
```bash
python final_transcript_system.py input_file.mp4
```

## Example Output

The script generates transcripts in this format:

```
Speaker 1: Good morning, honorable members. I would like to address the house regarding the budget allocation for education.

Speaker 2: Thank you, Mr. Speaker. I have a question about the proposed funding increase.

Speaker 1: The education budget has been increased by 15% compared to last year's allocation.

Speaker 3: [inaudible] ... concerns about implementation timeline.

Speaker 1: We expect the new funding to be available by the beginning of the next fiscal year.
```

## Supported formats

**Video**: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP, OGV
**Audio**: FLAC, MP3, WAV, M4A, AAC, OGG, WMA, Opus
