# About silero_tts_standalone
silero_tts_standalone is a simple script which can be used to TTS large text with [Silero TTS models](https://github.com/snakers4/silero-models) locally (do txt -> wav conversion).

By default, script is configured for Russian texts, but it can be reconfigured for any language supported by Silero models.

In order to work with non-Russian texts you should comment out spell_digits() function and its call in preprocess_text(), or (better) rewrite it with a module supporting your language.

The script was created to operate with large texts (over 1 MiB) but can handle small texts too.

It provides the following features:

* Basic text preprocessing (replace unsupported by model characters to supported, replace digits like 11 with "одиннадцать" to TTS them, limit line length according to punctuation)
* WAV file size limiting (WAV format is limited to 4 GiB file size) according to sentences (no awkward mid-word splits)
* Verbose run-time output with runtime estimation, full TTS size and length estimation and timestamps for each TTSed line

Usage:
   ./tts.py text.txt

The script was tested only with UTF-8 texts.

During runtime, it will output the following lines:

3/341 0:00:05/0:17:04 469/96065 chars 2/522 MiB 0:00:27/1:32:10 TTS 0:00:27@part0 0.5% : В ответ

* 3 - current line number
* 341 - total lines count
* 0:00:05 - elapsed time
* 0:17:04 - estimated time
* 469 - processed characters
* 96065 - total characters
* 2 - WAV size already written to output files (total)
* 522 - estimated WAV sizes (total)
* 0:00:27 - line timestamp (total)
* 1:32:10 - estimated length of all files
* 0:00:27 - line timestamp in current WAV file
* part0 - current WAV file number
* 0.5% - progress
* В ответ - processed string

Estimations may be inaccurate right after start, but after ~1 minute it will be more or less reliable.

Script will output the following files:

+ ${INPUT_FILENAME}\_preprocessed.txt - preprocessed text (it will be TTSed)
+ ${INPUT_FILENAME}\_0.wav
+ ${INPUT_FILENAME}\_1.wav
+ ... - TTS result

Requirements:

* Python 3.10.7+ (may work on earlier versions, but not tested)
* pytorch
* numpy
* [num2t4ru](https://github.com/seriyps/ru_number_to_text/tree/master/num2t4ru) (for spell_digits())

