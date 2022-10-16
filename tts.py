#!/usr/bin/env python3
# silero_tts_standalone
# Copyright (C) 2022  Soul Trace <S-trace@list.ru>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import re
import torch
import wave
import sys
from datetime import datetime, timedelta
from num2t4ru import num2text
from omegaconf import OmegaConf

# Configurable parameters:
model_id: str = 'v3_1_ru'
language: str = 'ru'
put_accent: bool = True
put_yo: bool = True
speaker: str = 'xenia'
sample_rate: int = 48000  # Hz - 48000, 24000 or 8000
torch_device: str = 'cpu'
torch_num_threads: int = 6  # Only effective for torch_device = 'cpu' - use 4-6 threads, larger count may slow down TTS
line_length_limits: dict = {
    'aidar': 882,
    'baya': 860,
    'eugene': 1000,
    'kseniya': 895,
    'xenia': 941,
    'random': 355,
}
wave_file_size_limit: int = 512 * 1024 * 1024  # 512 MiB - not more than 4GiB!
# 512 MiB ~= 1h 33m per file @48000, ~= 3h 6m per file @24000, ~= 9h 19m per file  @8000
# Exact formula:
# (512*1024*1024-wave_header_size)/wave_sample_width/wave_channels/sample_rate == wave_seconds

# Global constants - do not change:
wave_channels: int = 1  # Mono
wave_header_size: int = 44  # Bytes
wave_sample_width: int = int(16 / 8)  # 16 bits == 2 bytes


def main():
    print("main()")
    input_filename = process_args()
    origin_lines = load_file(input_filename)
    line_length_limit: int = line_length_limits[speaker]  # Max text length for speaker
    # del origin_lines
    preprocessed_lines, preprocessed_text_len = preprocess_text(origin_lines, line_length_limit)
    write_lines(input_filename + '_preprocessed.txt', preprocessed_lines)
    # exit(0)
    download_models_config()
    print_models_information()
    # find_max_line_length_all(input_filename, origin_lines)
    # exit(0)
    tts_model: torch.nn.Module = init_model(torch_device, torch_num_threads)
    print(f'Available speakers: {tts_model.speakers}')
    process_tts(tts_model, preprocessed_lines, input_filename, wave_file_size_limit, preprocessed_text_len)


def find_max_line_length_all(filename: str, lines: list):
    global speaker
    for speaker in line_length_limits.keys():
        find_max_line_length(filename, language, speaker, lines)


def find_max_line_length(filename: str, tts_language: str, tts_speaker: str, lines: list):
    new_length_limit: int = line_length_limits[tts_speaker]
    # preprocessed_lines, preprocessed_text_len = preprocess_text(origin_lines, line_length_limit)
    tts_model: torch.nn.Module = init_model(torch_device, torch_num_threads)  # Reinitialize model after speaker change
    print(f"Processing {tts_language}/{tts_speaker}")
    while True:
        try:
            print(f'Trying TTS with speaker {tts_speaker} and line_length_limit={new_length_limit}')
            preprocessed_lines, preprocessed_text_len = preprocess_text(lines, new_length_limit)
            process_tts(tts_model, preprocessed_lines, f"{filename}_{tts_speaker}", wave_file_size_limit,
                        preprocessed_text_len)
            break
        except Exception as exception:
            print(
                f'TTS failed with speaker {tts_speaker} and line_length_limit={new_length_limit} '
                f'with {type(exception)} exception: \n{exception}')
            new_length_limit -= 1
            print(f'Retrying with speaker {tts_speaker} and line_length_limit={new_length_limit}')
    print(F"Found limit: {tts_speaker} have line_length_limit={new_length_limit}")


def process_args() -> str:
    print("Processing args")
    if len(sys.argv) < 2:
        print(F"Usage: {sys.argv[0]} filename.txt")
        exit(1)
    input_filename: str = sys.argv[1]
    return input_filename


def load_file(filename: str) -> list:
    print("Loading file " + filename)
    with open(filename) as f:
        lines: list = f.readlines()
    f.close()
    return lines


def find_char_positions(string: str, char: str) -> list:
    pos: list = []  # list to store positions for each 'char' in 'string'
    for n in range(len(string)):
        if string[n] == char:
            pos.append(n)
    return pos


def find_max_char_position(positions: list, limit: int) -> int:
    max_position: int = 0
    for pos in positions:
        if pos < limit:
            max_position = pos
        else:
            break
    return max_position


def find_split_position(line: str, old_position: int, char: str, limit: int) -> int:
    positions: list = find_char_positions(line, char)
    new_position: int = find_max_char_position(positions, limit)
    position: int = max(new_position, old_position)
    return position


def spell_digits(line) -> str:
    digits: list = re.findall(r'\d+', line)
    # Sort digits from largest to smallest - else "1 11" will be "один один один" but not "один одиннадцать"
    digits = sorted(digits, key=len, reverse=True)
    for digit in digits:
        line = line.replace(digit, num2text(int(digit)))
    return line


def preprocess_text(lines: list, length_limit: int) -> (list, int):
    print(f"Preprocessing text with line length limit={length_limit}")

    if length_limit > 3:
        length_limit = length_limit - 2  # Keep a room for trailing char and '\n' char
    else:
        print(F"ERROR: line length limit must be >= 3, got {length_limit}")
        exit(1)

    preprocessed_text_len: int = 0
    preprocessed_lines: list = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces
        if line == '\n' or line == '':
            continue

        # Replace chars not supported by model
        line = line.replace("…", "...")  # Model does not handle "…"
        line = line.replace("*", " звёздочка ")
        line = spell_digits(line)

        # print("Processing line: " + line)
        while len(line) > 0:
            # v3_1_ru model does not handle long lines (over 990 chars)
            if len(line) < length_limit:
                # print("adding line: " + line)
                line = line + "\n"
                preprocessed_lines.append(line)
                preprocessed_text_len += len(line)
                break
            # Find position to split line between sentences
            split_position: int = 0
            split_position = find_split_position(line, split_position, ".", length_limit)
            split_position = find_split_position(line, split_position, "!", length_limit)
            split_position = find_split_position(line, split_position, "?", length_limit)

            # If no punctuation found - try to split on space
            if split_position == 0:
                split_position = find_split_position(line, split_position, " ", length_limit)

            # If no punctuation found - force split at limit
            if split_position == 0:
                split_position = length_limit

            # Keep trailing char, add newline
            part: str = line[0:split_position + 1] + "\n"
            # print(F'Line too long - splitting at position {split_position}:  {line}')
            preprocessed_lines.append(part)
            preprocessed_text_len += len(part)
            # Skip trailing char from previous part
            line = line[split_position + 1:]
            # print ("Rest of line: " + line)
    return preprocessed_lines, preprocessed_text_len


def write_lines(filename: str, lines: list):
    print("Writing file " + filename)
    with open(filename, 'w') as f:
        f.writelines(lines)
        f.close()


def print_models_information():
    config = OmegaConf.load('latest_silero_models.yml')
    available_languages = list(config.tts_models.keys())
    print(f'Available languages {available_languages}')
    for lang in available_languages:
        models: list = list(config.tts_models.get(lang).keys())
        print(f'Available models for {lang}: {models}')


def download_models_config():
    print("Downloading models config")
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                   'latest_silero_models.yml',
                                   progress=False)


def init_model(device: str, threads_count: int) -> torch.nn.Module:
    print("Initialising model")
    device: torch.device = torch.device(device)
    torch.set_num_threads(threads_count)
    tts_model, tts_sample_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                model='silero_tts',
                                                language=language,
                                                speaker=model_id)
    tts_model.to(device)  # gpu or cpu
    return tts_model


def init_wave_file(name: str, channels: int, sample_width: int, rate: int):
    print(f'Initialising wave file {name} with {channels} channels {sample_width} sample width {rate} sample rate')
    wf = wave.open(name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    return wf


class Stats:
    def __init__(self, preprocessed_text_len: int):
        self.start_time = int(datetime.now().timestamp())
        self.preprocessed_text_len = preprocessed_text_len

    preprocessed_text_len: int
    processed_text_len: int = 0
    done_percent: float = 0
    start_time: int
    run_time: str = "0:00:00"
    run_time_est: str = "0:00:00"
    wave_data_current: int = 0
    wave_data_total: int = 0
    wave_mib: int = 0
    wave_mib_est: int = 0
    tts_time: str = "0:00:00"
    tts_time_est: str = "0:00:00"
    tts_time_current: str = "0:00:00"

    def update(self, line: str, next_chunk_size: int):
        self.wave_data_total += next_chunk_size
        self.wave_data_current += next_chunk_size
        self.processed_text_len += len(line)
        # Percentage calculation
        self.done_percent = round(self.processed_text_len * 100 / self.preprocessed_text_len, 1)
        # Wave size estimation
        self.wave_mib = int((self.wave_data_total / 1024 / 1024))
        self.wave_mib_est = int(
            (self.wave_data_total / 1024 / 1024 * self.preprocessed_text_len / self.processed_text_len))
        # Run time estimation
        current_time: int = int(datetime.now().timestamp())
        run_time_s: int = current_time - self.start_time
        run_time_est_s: int = int(run_time_s * self.preprocessed_text_len / self.processed_text_len)
        self.run_time = str(timedelta(seconds=run_time_s))
        self.run_time_est = str(timedelta(seconds=run_time_est_s))
        # TTS time estimation
        tts_time_s: int = int((self.wave_data_total / wave_channels / wave_sample_width / sample_rate))
        tts_time_est_s: int = int((tts_time_s * self.preprocessed_text_len / self.processed_text_len))
        self.tts_time = str(timedelta(seconds=tts_time_s))
        self.tts_time_est = str(timedelta(seconds=tts_time_est_s))
        tts_time_current_s: int = int((self.wave_data_current / wave_channels / wave_sample_width / sample_rate))
        self.tts_time_current = str(timedelta(seconds=tts_time_current_s))

    def next_file(self):
        self.wave_data_current = 0


def write_wave_chunk(wf, audio, audio_size: int, filename: str, wave_data_limit: int, wave_file_number: int,
                     stats: Stats):
    next_chunk_size = int(audio.size()[0] * wave_sample_width)
    if audio_size + next_chunk_size > wave_data_limit:
        print(F"Wave written {audio_size} limit={wave_data_limit} - creating new wave!")
        wf.close()
        stats.next_file()
        wave_file_number += 1
        audio_size = wave_header_size + next_chunk_size
        wf = init_wave_file(F'{filename}_{wave_file_number}.wav',
                            wave_channels, wave_sample_width, sample_rate)
    else:
        audio_size += next_chunk_size
        wf.writeframes((audio * 32767).numpy().astype('int16'))
    return wf, audio_size, wave_file_number


# Process TTS for preprocessed_lines
def process_tts(tts_model: torch.nn.Module, lines: list, output_filename: str, wave_data_limit: int,
                preprocessed_text_len: int):
    print("Starting TTS")
    s = Stats(preprocessed_text_len)
    current_line: int = 0
    audio_size: int = wave_header_size
    wave_file_number: int = 0
    next_chunk_size: int
    wf = init_wave_file(F'{output_filename}_{wave_file_number}.wav', wave_channels, wave_sample_width, sample_rate)
    for line in lines:
        if line == '\n' or line == '':
            continue
        print(
            F'{current_line}/{len(lines)} {s.run_time}/{s.run_time_est} '
            F'{s.processed_text_len}/{s.preprocessed_text_len} chars '
            F'{s.wave_mib}/{s.wave_mib_est} MiB {s.tts_time}/{s.tts_time_est} TTS '
            F'{s.tts_time_current}@part{wave_file_number} {s.done_percent}% : {line}'
        )
        try:
            audio = tts_model.apply_tts(text=line,
                                        speaker=speaker,
                                        sample_rate=sample_rate,
                                        put_accent=put_accent,
                                        put_yo=put_yo)
            next_chunk_size = int(audio.size()[0] * wave_sample_width)
            wf, audio_size, wave_file_number = write_wave_chunk(wf, audio, audio_size, output_filename,
                                                                wave_data_limit, wave_file_number, s)
        except ValueError:
            print("TTS failed!")
            next_chunk_size = 0

        current_line += 1
        s.update(line, next_chunk_size)


main()
