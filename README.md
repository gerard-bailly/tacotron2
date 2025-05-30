# Tacotron2
PyTorch implementation of Tacotron 2 with several (optional) extensions:
1. Multiple decoders
2. Mixed text/phones input
3. Phonetic predictor
4. Speaker & style embeddings
5. Pre-processed target files with direct access (avoid spliting audiobooks in thousands of utterances)
6. Reference encoder
7. Freezing layers
8. Yaml configuration file. Examples could be found at: 
[https://zenodo.org/records/14893481](https://zenodo.org/records/14893481/files/tc2_italian.yaml?download=1) for Italian and [https://zenodo.org/records/7560290](https://zenodo.org/api/records/13903548/draft/files/tc2.yaml) for French
  
## Pre-processing target files
1. Frames (Mel-spectrograms, action units...) should be stored and will be generated in the following format:
  - Header with 4 int32 values: (nb-of-frames, nb-parameters; numerator of sampling frequency; denominator of sampling frequency); Note that waveglow samples spectrograms of 22050 Hz audio signals at 22050/256=86.1328125 Hz
  - Followed by nb-of-frames frames of nb-parameters float32 values
  - Naming recommendations: <author>_<book>_<reader>_<style>_<volume>_<chapter>.<parameter_name>
  - Note that <reader>, <style> and <parameter_name> are used in the Yaml configuration file to automatically select the appropriate items in the lists of keys 'speakers', 'styles' and 'ext_data'
2. A .csv file describing utterances. Each line contains fields separated by "|"
  - They should contain at least 4 fields: <target_file>|<start ms>|<end ms>|<text or input phones separated by spaces in {}
  -An additional field may specify aligned output phones separated by spaces
  - The key 'lgs_sil_add' in the Yaml configuration file specifies how many seconds of ambient silence (typically 0.1s) are added before <start ms> and <end ms>. Input text entries should "explain" these silences: we recommend to begin and end utterances produced in isolation with the end-of-chapter symbol "§", otherwise to start the current utterance with the final punctuation of the previous utterance.
  - Examples could be found at: 
[https://zenodo.org/records/7560290](https://zenodo.org/records/7560290/files/AD_train.csv?download=1) for French and 
[https://zenodo.org/records/14893481](https://zenodo.org/records/14893481/files/IT.csv?download=1) for Italian

3. Language-specific lists of text characters, input phones & output phones are specified in def_symbols.py respectively by _specific_characters, valid_symbols & valid_alignments
  - Language is selected in the Yaml configuration file via the key 'language'

## Training
1. python3 do_train.py --output_directory <...> -c tacotron2_* --config tc2.yaml --hparams "{factor_pho: 1.00, nb_epochs: 10, learning_rate: 0.0002, batch_size: 40, nm_csv_train: '<...>.csv', lgs_max: 10}"
2. Pre-trained models can be found at [https://zenodo.org/records/14893481](https://zenodo.org/records/14893481/files/tacotron2_IT?download=1) for Italian and [https://zenodo.org/records/7560290](https://zenodo.org/records/13903548/files/tacotron2_ALL?download=1) for French

## Batch inference/synthesis
1. python3 do_syn.py --output_directory <...> --vocoder=waveglow_NEB.pt --tacotron tacotron2_FR -e '' --config tc2.yaml --hparams "{nm_csv_test: '<...>.csv'}"
  - The list of supported neural vocoders are listed in the key 'vocoder' in the Yaml configuration file: for now, 'waveglow' and 'hifigan' are supported

## On-line Inference/synthesis
1. python3 do_tts.py --silent --no_auto_numbering --play_wav --speaker <spk>  %% On-line TTS with config tc2.yaml, play, speaker <spk> and WAVEGLOW vocoder by default
Your input>> ...
  - to use HIFIGAN: -v hifigan/generator_universal.pth.tar

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based Generative Network for Speech Synthesis

[HiFi_GAN](https://github.com/jik876/hifi-gan) GAN-based model capable of generating high fidelity speech efficiently
