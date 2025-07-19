import sys
import os
import base64
import time
import subprocess
import google.generativeai as genai
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")

genai.configure(api_key= API_KEY)

UNWANTED_PHRASES = [
    "Okay, here is the transcription of the audio you provided:",
    "Here is the transcription of the audio you provided:",
    "Here's the transcription of the audio you provided:",
    "Here is the transcription:",
    "Here's the transcription:",
    "Transcription:",
    "Here is the audio transcription:",
    "Here's the audio transcription:",
    "Audio transcription:",
    "Parliamentary audio transcription:",
    "Here is the parliamentary audio transcription:",
    "Here's the parliamentary audio transcription:",
    "The speakers in this audio are:",
    "Speakers identified:",
    "In this audio, we have:",
    "The following speakers are present:"
]

def clean_transcript(transcript):
    for phrase in UNWANTED_PHRASES:
        if transcript.startswith(phrase):
            transcript = transcript[len(phrase):].strip()
            break
    return transcript.strip()

def cleanup_chunk_files(chunk_files):
    deleted_count = 0
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
            deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not delete {chunk_file}: {e}")
    
    return deleted_count

def print_statistics(chunk_files, all_transcripts, total_time, output_file):
    successful_chunks = sum(1 for t in all_transcripts if "TRANSCRIPTION FAILED" not in t)
    success_rate = (successful_chunks / len(chunk_files)) * 100
    
    print(f"Transcription completed successfully")
    print(f"Output saved to: {output_file}")
    print(f"Statistics:")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Total time: {total_time/60:.1f} minutes")

def calculate_progress(i, total_chunks, start_time, progress_interval=5):
    """Calculate and print progress"""
    if i % progress_interval == 0 or i == total_chunks:
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (total_chunks - i) * avg_time
        print(f"Progress: {i}/{total_chunks} ({i/total_chunks*100:.1f}%) - Est. remaining: {remaining/60:.1f} min")

def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()

def get_base_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def split_audio(input_file, output_dir="chunks", duration=60, overlap=10):
    if not os.path.exists(input_file):
        return []
    os.makedirs(output_dir, exist_ok=True)
    cmd_duration = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", input_file
    ]
    try:
        result = subprocess.run(cmd_duration, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return []
    effective_duration = duration - overlap
    num_chunks = int(total_duration // effective_duration) + 1
    chunk_info = []  
    for i in range(num_chunks):
        start_time = i * effective_duration
        if start_time >= total_duration:
            break
        if i == num_chunks - 1:
            chunk_duration = total_duration - start_time
        else:
            chunk_duration = duration
        output_file = os.path.join(output_dir, f"chunk_{i+1:02d}_{duration}s.flac")
        cmd = [
            "ffmpeg", "-i", input_file,
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-c:a", "flac",
            "-y", output_file
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_info.append({
                'file': output_file,
                'start_time': start_time,
                'duration': chunk_duration
            })
        except subprocess.CalledProcessError as e:
            continue
    return chunk_info

def encode_audio_file(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
    return base64.b64encode(audio_data).decode('utf-8')

def transcribe_chunk(audio_path, previous_context="", chunk_number=1, total_chunks=1, start_time=0, end_time=0):    
    time_range = format_time_range(start_time, end_time)
    print(f"Transcribing {os.path.basename(audio_path)} (chunk {chunk_number}/{total_chunks}) - {time_range}...")
    
    file_size = os.path.getsize(audio_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    audio_base64 = encode_audio_file(audio_path)
    
    context_info = ""
    if previous_context:
        lines = previous_context.strip().split('\n')
        last_lines = lines[-3:] if len(lines) >= 3 else lines
        context_info = f"\n\nPrevious context (last few lines):\n" + '\n'.join(last_lines) + "\n\nContinue the transcription from where it left off:"
    
    prompt = f"""Transcribe this parliamentary audio accurately. Format as:
[Speaker Name]: [transcript text]
[Speaker Name]: [transcript text]

Requirements:
- IMPORTANT: Try to identify actual speaker names from the audio content (e.g., when speakers introduce themselves, are addressed by name, or mention their titles/roles)
- Listen for names being mentioned like "Mr. Smith", "Ms. Johnson", "The Minister", "The Chairman", etc.
- If speakers address each other by name, use those names consistently
- For parliamentary settings, listen for titles like "Minister", "Chairman", "Member", followed by names
- If you can identify actual names or titles, use them instead of generic speaker labels
- Only use "Speaker 1", "Speaker 2" as fallback when no names can be determined
- If you cannot identify speakers clearly at all, use "Unknown Speaker"
- Provide accurate transcription with proper punctuation
- Handle parliamentary terminology and formal language correctly
- Maintain the flow and context of the conversation
- Correct any obvious misheard words based on context
- If audio is unclear or inaudible, mark as "[inaudible]" or "[unclear]"
- Do not repeat words or phrases unnecessarily
- Stop transcription if audio becomes completely unclear
- Maintain speaker consistency with previous context - if a speaker was identified by name in previous context, continue using that name{context_info}

Start directly with the transcription without any introduction or acknowledgment."""
    
    try:
        content = [
            prompt,
            {
                "mime_type": "audio/flac",
                "data": audio_base64
            }
        ]
        
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  
                top_k=1,
                top_p=0.8,
                max_output_tokens=8192,
            )
        )
        
        if response.text:
            transcript = response.text.strip()
            transcript = clean_transcript(transcript)
            
            time_range_header = f"{time_range}\n"
            transcript = time_range_header + transcript
            
            return transcript
        else:
            return ""
            
    except Exception as e:
        return ""


def extract_audio_from_video(video_file, output_dir="."):    
    if not os.path.exists(video_file):
        return None
    
    base_name = get_base_name(video_file)
    audio_file = os.path.join(output_dir, f"{base_name}_audio.flac")
    
    cmd = [
        "ffmpeg", "-i", video_file,
        "-vn",
        "-acodec", "flac",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_file
        
    except subprocess.CalledProcessError as e:
        return None

def deduplicate_overlapping_content(transcripts, overlap_seconds=10):    
    if len(transcripts) <= 1:
        return transcripts
    
    cleaned_transcripts = []
    previous_end = ""
    
    for i, transcript in enumerate(transcripts):
        lines = transcript.strip().split('\n')
        
        time_range_header = ""
        content_lines = lines
        if lines and lines[0].startswith('[') and ' - ' in lines[0]:
            time_range_header = lines[0]
            content_lines = lines[1:]
        
        if i == 0:
            cleaned_transcripts.append(transcript)
            if content_lines:
                last_line = content_lines[-1]
                words = last_line.split()
                previous_end = ' '.join(words[-10:]) if len(words) >= 10 else last_line
        else:
            current_start = ""
            if content_lines:
                first_line = content_lines[0]
                words = first_line.split()
                current_start = ' '.join(words[:10]) if len(words) >= 10 else first_line
            
            if previous_end and current_start and len(previous_end) > 20:
                if current_start.lower() in previous_end.lower() or previous_end.lower() in current_start.lower():
                    overlap_found = False
                    cleaned_lines = []
                    for line in content_lines:
                        line_words = line.split()
                        if len(line_words) >= 5:
                            line_start = ' '.join(line_words[:5])
                            if line_start.lower() not in previous_end.lower():
                                overlap_found = True
                        if overlap_found:
                            cleaned_lines.append(line)
                    if cleaned_lines:
                        if time_range_header:
                            cleaned_transcript = time_range_header + '\n' + '\n'.join(cleaned_lines)
                        else:
                            cleaned_transcript = '\n'.join(cleaned_lines)
                        cleaned_transcripts.append(cleaned_transcript)
                    else:
                        continue
                else:
                    cleaned_transcripts.append(transcript)
            else:
                cleaned_transcripts.append(transcript)
            
            if cleaned_transcripts:
                last_transcript = cleaned_transcripts[-1]
                transcript_lines = last_transcript.strip().split('\n')
                content_lines = transcript_lines
                if transcript_lines and transcript_lines[0].startswith('[') and ' - ' in transcript_lines[0]:
                    content_lines = transcript_lines[1:]
                
                if content_lines:
                    last_line = content_lines[-1]
                    words = last_line.split()
                    previous_end = ' '.join(words[-10:]) if len(words) >= 10 else last_line
    
    return cleaned_transcripts

def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
    return get_file_extension(file_path) in video_extensions

def is_audio_file(file_path):
    audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.opus'}
    return get_file_extension(file_path) in audio_extensions


def batch_transcribe(audio_file, output_file="full_transcript.txt"):    
    print(f"Input file: {audio_file}")
    print(f"Output file: {output_file}")
    
    chunk_info = split_audio(audio_file, duration=90, overlap=5)
    
    if not chunk_info:
        print("No chunks created. Exiting.")
        return
    
    start_time = time.time()
    
    results = [None] * len(chunk_info)
    
    def process_chunk_with_context(chunk_data_info):
        index, chunk_data = chunk_data_info
        chunk_file = chunk_data['file']
        chunk_start_time = chunk_data['start_time']
        chunk_end_time = chunk_data['start_time'] + chunk_data['duration']
        
        context = ""
        speaker_context = ""
        
        if index > 0 and results[index-1]:
            prev_transcript = results[index-1]
            lines = prev_transcript.strip().split('\n')
            if lines:
                context_lines = lines[-3:] if len(lines) >= 3 else lines
                context = '\n'.join(context_lines)
            
            completed_transcripts = [r for r in results[:index] if r]
            speaker_context = build_speaker_context(completed_transcripts)
        
        full_context = context + speaker_context
        
        result = transcribe_chunk(
            chunk_file, 
            previous_context=full_context,
            chunk_number=index + 1,
            total_chunks=len(chunk_info),
            start_time=chunk_start_time,
            end_time=chunk_end_time
        )
        return index, result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_chunk = {
            executor.submit(process_chunk_with_context, (i, chunk_data)): i 
            for i, chunk_data in enumerate(chunk_info)
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index, transcript = future.result()
            results[chunk_index] = transcript
            completed += 1
            
            calculate_progress(completed, len(chunk_info), start_time)
    
    all_transcripts = []
    for i, transcript in enumerate(results):
        if transcript:
            all_transcripts.append(transcript + "\n")
        else:
            all_transcripts.append("[TRANSCRIPTION FAILED]\n")
    
    if all_transcripts:
        deduplicated_transcripts = deduplicate_overlapping_content(all_transcripts, overlap_seconds=8)
        
        full_transcript = "".join(deduplicated_transcripts)
        
        full_transcript = standardize_speaker_names(full_transcript)
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(full_transcript)
        
        chunk_files = [chunk_data['file'] for chunk_data in chunk_info]
        cleanup_chunk_files(chunk_files)
        
        total_time = time.time() - start_time
        print_statistics(chunk_files, all_transcripts, total_time, output_file)
        
    else:
        print("No transcripts were generated")

def extract_speaker_names(transcript):
    speaker_names = set()
    lines = transcript.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('[') and ' - ' in line:
            continue
        if ':' in line:
            speaker_part = line.split(':', 1)[0].strip()
            speaker_part = speaker_part.replace('[', '').replace(']', '')
            
            if not any(generic in speaker_part.lower() for generic in ['speaker', 'unknown']):
                if len(speaker_part) > 0 and len(speaker_part) < 50:
                    speaker_names.add(speaker_part)
    
    return list(speaker_names)

def build_speaker_context(previous_transcripts):
    all_speakers = set()
    for transcript in previous_transcripts:
        speakers = extract_speaker_names(transcript)
        all_speakers.update(speakers)
    
    if all_speakers:
        speaker_list = list(all_speakers)
        context = f"\n\nPreviously identified speakers: {', '.join(speaker_list)}"
        context += "\nPlease maintain consistency with these speaker names when they appear again."
        return context
    return ""

def standardize_speaker_names(transcript):
    lines = transcript.split('\n')
    speaker_mapping = {}
    
    speaker_variations = {}
    for line in lines:
        line = line.strip()
        if line.startswith('[') and ' - ' in line:
            continue
        if ':' in line:
            speaker_part = line.split(':', 1)[0].strip()
            speaker_part = speaker_part.replace('[', '').replace(']', '')
            
            base_name = speaker_part.lower()
            
            base_name = base_name.replace('mr.', '').replace('ms.', '').replace('mrs.', '')
            base_name = base_name.replace('dr.', '').replace('prof.', '').replace('minister', '')
            base_name = base_name.replace('chairman', '').replace('chairwoman', '').strip()
            
            if base_name and len(base_name) > 1:
                if base_name not in speaker_variations:
                    speaker_variations[base_name] = []
                speaker_variations[base_name].append(speaker_part)
    
    for base_name, variations in speaker_variations.items():
        if len(variations) > 1:
            best_version = max(variations, key=len)
            for variation in variations:
                speaker_mapping[variation] = best_version
    
    standardized_lines = []
    for line in lines:
        if line.strip().startswith('[') and ' - ' in line.strip():
            standardized_lines.append(line)
        elif ':' in line:
            speaker_part, content = line.split(':', 1)
            speaker_part = speaker_part.strip().replace('[', '').replace(']', '')
            
            if speaker_part in speaker_mapping:
                speaker_part = speaker_mapping[speaker_part]
            
            standardized_lines.append(f"{speaker_part}:{content}")
        else:
            standardized_lines.append(line)
    
    return '\n'.join(standardized_lines)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def format_time_range(start_seconds, end_seconds):
    start_time = format_timestamp(start_seconds)
    end_time = format_timestamp(end_seconds)
    return f"[{start_time} - {end_time}]"

def main():
    if len(sys.argv) < 2:
        print("Usage: python final_transcript_system.py <audio_or_video_file> [output_file]")
        return
    
    input_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        base_name = get_base_name(input_file)
        output_file = f"{base_name}_transcript.txt"
    
    if not os.path.exists(input_file):
        return
    
    if is_video_file(input_file):
        print("Video file detected. Extracting audio first...")
        audio_file = extract_audio_from_video(input_file)
        if not audio_file:
            print("Failed to extract audio from video. Exiting.")
            return
        print(f"Using extracted audio: {audio_file}")
        batch_transcribe(audio_file, output_file)
        
    elif is_audio_file(input_file):
        print("Audio file detected. Processing directly...")
        batch_transcribe(input_file, output_file)
        
    else:
        print(f"Unsupported file format: {input_file}")
        return

if __name__ == "__main__":
    main()