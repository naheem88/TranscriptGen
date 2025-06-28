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
    "Here's the parliamentary audio transcription:"
]

def clean_transcript(transcript):
    """Remove unwanted phrases from transcript"""
    for phrase in UNWANTED_PHRASES:
        if transcript.startswith(phrase):
            transcript = transcript[len(phrase):].strip()
            break
    return transcript.strip()

def cleanup_chunk_files(chunk_files):
    """Clean up temporary chunk files"""
    deleted_count = 0
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
            deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not delete {chunk_file}: {e}")
    
    return deleted_count

def print_statistics(chunk_files, all_transcripts, total_time, output_file):
    """Print transcription statistics"""
    successful_chunks = sum(1 for t in all_transcripts if "TRANSCRIPTION FAILED" not in t)
    success_rate = (successful_chunks / len(chunk_files)) * 100
    full_transcript = "".join(all_transcripts)
    
    print(f"Transcription completed successfully")
    print(f"Output saved to: {output_file}")
    print(f"Statistics:")
    print(f"   - Total chunks: {len(chunk_files)}")
    print(f"   - Successful: {successful_chunks}")
    print(f"   - Failed: {len(chunk_files) - successful_chunks}")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Total time: {total_time/60:.1f} minutes")
    print(f"   - Average time per chunk: {total_time/len(chunk_files):.1f} seconds")

def calculate_progress(i, total_chunks, start_time, progress_interval=5):
    """Calculate and print progress"""
    if i % progress_interval == 0 or i == total_chunks:
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (total_chunks - i) * avg_time
        print(f"Progress: {i}/{total_chunks} ({i/total_chunks*100:.1f}%) - Est. remaining: {remaining/60:.1f} min")

def get_file_extension(file_path):
    """Get file extension in lowercase"""
    return os.path.splitext(file_path)[1].lower()

def get_base_name(file_path):
    """Get base name without extension"""
    return os.path.splitext(os.path.basename(file_path))[0]

def split_audio(input_file, output_dir="chunks", duration=60, overlap=10):
    """Split audio file into overlapping chunks for better context preservation"""
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
    chunk_files = []
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
            chunk_files.append(output_file)
        except subprocess.CalledProcessError as e:
            continue
    return chunk_files

def encode_audio_file(audio_path):
    """Encode audio file to base64"""
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
    return base64.b64encode(audio_data).decode('utf-8')

def transcribe_chunk(audio_path, previous_context="", chunk_number=1, total_chunks=1):
    """Transcribe a single audio chunk using Gemini 2.0 Flash with context preservation"""
    
    print(f"Transcribing {os.path.basename(audio_path)} (chunk {chunk_number}/{total_chunks})...")
    
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
Speaker X: [transcript text]
Speaker Y: [transcript text]

Requirements:
- Identify and label different speakers (Speaker 1, Speaker 2, etc.)
- Provide accurate transcription with proper punctuation
- Handle parliamentary terminology and formal language correctly
- Maintain the flow and context of the conversation
- Correct any obvious misheard words based on context
- If you cannot identify speakers clearly, use "Unknown Speaker"
- If audio is unclear or inaudible, mark as "[inaudible]" or "[unclear]"
- Do not repeat words or phrases unnecessarily
- Stop transcription if audio becomes completely unclear
- Maintain speaker consistency with previous context{context_info}

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
            return transcript
        else:
            return ""
            
    except Exception as e:
        return ""

def batch_transcribe(audio_file, output_file="full_transcript.txt"):
    """Complete transcription pipeline with context-aware processing"""
    
    print(f"Input file: {audio_file}")
    print(f"Output file: {output_file}")
    
    # Step 1: Split audio into overlapping chunks
    chunk_files = split_audio(audio_file, overlap=10)
    
    if not chunk_files:
        print("No chunks created. Exiting.")
        return
    
    # Step 2: Transcribe chunks with context preservation
    all_transcripts = []
    accumulated_context = ""
    start_time = time.time()
    
    for i, chunk_file in enumerate(chunk_files, 1):
        # Pass previous context to maintain continuity
        transcript = transcribe_chunk(
            chunk_file, 
            previous_context=accumulated_context,
            chunk_number=i,
            total_chunks=len(chunk_files)
        )
        
        if transcript:
            all_transcripts.append(transcript + "\n")
            # Update context for next chunk (keep last few lines)
            accumulated_context += transcript + "\n"
            # Keep only last 500 characters to avoid context overflow
            if len(accumulated_context) > 500:
                lines = accumulated_context.split('\n')
                accumulated_context = '\n'.join(lines[-5:])  # Keep last 5 lines
        else:
            all_transcripts.append("[TRANSCRIPTION FAILED]\n")
        
        calculate_progress(i, len(chunk_files), start_time, 5)
    
    # Step 3: Combine and save
    if all_transcripts:
        # Step 3a: Deduplicate overlapping content
        deduplicated_transcripts = deduplicate_overlapping_content(all_transcripts, overlap_seconds=10)
        
        # Step 3b: Combine final transcripts
        full_transcript = "".join(deduplicated_transcripts)
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(full_transcript)
        
        # Step 4: Clean up chunk files
        cleanup_chunk_files(chunk_files)
        
        # Statistics
        total_time = time.time() - start_time
        print_statistics(chunk_files, all_transcripts, total_time, output_file)
        
    else:
        print("No transcripts were generated")

def extract_audio_from_video(video_file, output_dir="."):
    """Extract audio from video file to FLAC format"""
    
    if not os.path.exists(video_file):
        return None
    
    # Generate output filename
    base_name = get_base_name(video_file)
    audio_file = os.path.join(output_dir, f"{base_name}_audio.flac")
    
    cmd = [
        "ffmpeg", "-i", video_file,
        "-vn",  # No video
        "-acodec", "flac",  # FLAC audio codec
        "-ar", "16000",  # 16kHz sample rate (good for speech)
        "-ac", "1",  # Mono audio
        "-y",  # Overwrite output file
        audio_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_file
        
    except subprocess.CalledProcessError as e:
        return None

def deduplicate_overlapping_content(transcripts, overlap_seconds=10):
    """Remove duplicate content from overlapping chunks"""
    
    if len(transcripts) <= 1:
        return transcripts
    
    cleaned_transcripts = []
    previous_end = ""
    
    for i, transcript in enumerate(transcripts):
        if i == 0:
            # First chunk - keep as is
            cleaned_transcripts.append(transcript)
            # Extract last few words for next comparison
            lines = transcript.strip().split('\n')
            if lines:
                last_line = lines[-1]
                words = last_line.split()
                previous_end = ' '.join(words[-10:]) if len(words) >= 10 else last_line
        else:
            # Check for overlap with previous chunk
            current_start = ""
            lines = transcript.strip().split('\n')
            if lines:
                first_line = lines[0]
                words = first_line.split()
                current_start = ' '.join(words[:10]) if len(words) >= 10 else first_line
            
            if previous_end and current_start and len(previous_end) > 20:
                if current_start.lower() in previous_end.lower() or previous_end.lower() in current_start.lower():
                    overlap_found = False
                    cleaned_lines = []
                    for line in lines:
                        line_words = line.split()
                        if len(line_words) >= 5:
                            line_start = ' '.join(line_words[:5])
                            if line_start.lower() not in previous_end.lower():
                                overlap_found = True
                        if overlap_found:
                            cleaned_lines.append(line)
                    if cleaned_lines:
                        cleaned_transcript = '\n'.join(cleaned_lines)
                        cleaned_transcripts.append(cleaned_transcript)
                    else:
                        continue
                else:
                    cleaned_transcripts.append(transcript)
            else:
                cleaned_transcripts.append(transcript)
            
            # Update previous_end for next iteration
            if cleaned_transcripts:
                last_transcript = cleaned_transcripts[-1]
                lines = last_transcript.strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    words = last_line.split()
                    previous_end = ' '.join(words[-10:]) if len(words) >= 10 else last_line
    
    return cleaned_transcripts

def is_video_file(file_path):
    """Check if file is a video file based on extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
    return get_file_extension(file_path) in video_extensions

def is_audio_file(file_path):
    """Check if file is an audio file based on extension"""
    audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.opus'}
    return get_file_extension(file_path) in audio_extensions


def batch_transcribe_optimized(audio_file, output_file="full_transcript.txt"):
    """Optimized transcription pipeline with quality and speed"""
    
    print(f"Input file: {audio_file}")
    print(f"Output file: {output_file}")
    
    # Step 1: Split audio into optimized chunks
    chunk_files = split_audio(audio_file, duration=90, overlap=5)  # Optimized for speed + quality
    
    if not chunk_files:
        print("No chunks created. Exiting.")
        return
    
    # Step 2: Parallel transcription with context preservation
    start_time = time.time()
    
    # Create a list to store results in order
    results = [None] * len(chunk_files)
    
    def process_chunk_with_context(chunk_info):
        index, chunk_file = chunk_info
        
        # Get context from previous chunks if available
        context = ""
        if index > 0 and results[index-1]:
            # Extract last few lines from previous chunk for context
            prev_transcript = results[index-1]
            lines = prev_transcript.strip().split('\n')
            if lines:
                context_lines = lines[-3:] if len(lines) >= 3 else lines
                context = '\n'.join(context_lines)
        
        result = transcribe_chunk(
            chunk_file, 
            previous_context=context,
            chunk_number=index + 1,
            total_chunks=len(chunk_files)
        )
        return index, result
    
    # Process chunks in parallel (limited to 3 workers to maintain quality)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk_with_context, (i, chunk_file)): i 
            for i, chunk_file in enumerate(chunk_files)
        }
        
        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index, transcript = future.result()
            results[chunk_index] = transcript
            completed += 1
            
            # Progress update
            calculate_progress(completed, len(chunk_files), start_time)
    
    # Step 3: Combine and save
    all_transcripts = []
    for i, transcript in enumerate(results):
        if transcript:
            all_transcripts.append(transcript + "\n")
        else:
            all_transcripts.append("[TRANSCRIPTION FAILED]\n")
    
    if all_transcripts:
        # Deduplicate overlapping content
        deduplicated_transcripts = deduplicate_overlapping_content(all_transcripts, overlap_seconds=8)
        
        full_transcript = "".join(deduplicated_transcripts)
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(full_transcript)
        
        # Clean up
        cleanup_chunk_files(chunk_files)
        
        # Statistics
        total_time = time.time() - start_time
        print_statistics(chunk_files, all_transcripts, total_time, output_file)
        
    else:
        print("No transcripts were generated")

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
    
    # Determine file type and process accordingly
    if is_video_file(input_file):
        print("Video file detected. Extracting audio first...")
        audio_file = extract_audio_from_video(input_file)
        if not audio_file:
            print("Failed to extract audio from video. Exiting.")
            return
        print(f"Using extracted audio: {audio_file}")
        batch_transcribe_optimized(audio_file, output_file)
        
    elif is_audio_file(input_file):
        print("Audio file detected. Processing directly...")
        batch_transcribe_optimized(input_file, output_file)
        
    else:
        print(f"Unsupported file format: {input_file}")
        return

if __name__ == "__main__":
    main() 