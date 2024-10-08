# tacotron2
PyTorch implementation of Tacotron 2 with several (optional) extensions:
1. Multiple decoders
2. Mixed text/phones input
3. Phonetic predictor
4. Speaker & style embeddings
5. Pre-processed target files with direct access (avoid spliting audiobooks in thousands of utterances)
6. Yaml configuration file

## Pre-processing target files
1. Mel-spectrograms, frames of action units should be stored and will be generated in the following format:
  a. Header with 4 int32 values: (nb-of-frames, nb-parameters; numerator of sampling frequency; denominator of sampling frequency);
  b. Followed by nb-of-frames frames of nb-parameters float32 values
  c. Naming recommendations: <author>_<book>_<reader>_<style>_<volume>_<chapter>.<parameter_name>
2. A .csv file describing utterances. Each line contains fields separated by "|".
  a. They should contain at least 4 fields: <target_file>|<start ms>|<end ms>|<text or input phones separated by spaces in {}>
  b. An additional field may specify aligned output phones separated by spaces
3. Language-specific lists of text characters, input phones and output phones are specified in def_


