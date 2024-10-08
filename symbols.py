import re
global symbols, out_symbols

def init_symbols(hps):

	if hps['language']=='french':
			valid_symbols = [
				'a', 'a~', 'b', 'd', 'e', 'e^', 'e~', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ng', 'n~', 'o', 'o^', 'o~',
				'p', 'q', 'r', 's', 's^', 't', 'u', 'v', 'w', 'x', 'x^', 'x~', 'y', 'z', 'z^'
			]
			valid_alignments = [
				'_','a','a&i','a&j','a~','b','b&q','d','d&q','d&z','d&z^','e','e^','e~','f','f&q','g','g&q','g&z','h','i','j','j&i',
				'j&u','j&q','i&j','k','k&q','k&s','k&s&q','l','l&q','m','m&q','n','n&q','ng','o','o^','o~','p','q','r',
				'r&w','r&q','s','s&q','s^','t','t&q','t&s','t&s^','u','v','w','w&a','x','x^','x~','y','z','z&q','z^','n~','__','p&q'
			]
			_specific_characters = '[]§«»ÀÂÇÉÊÎÔàâæçèéêëîïôùûü¬~"' # GB: new symbols for turntaking & ldots, [] are for notes, " for new terms.
   elif hps['language']=='italian':
			valid_symbols = [
				'a',  'b', 'd', 'e',  'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ng', 'o', 
				'p', 'q', 'r', 's', 's^', 't', 't&s^', 'u', 'v', 'w', 'x', 'y', 'z', 'z^'
			]
			valid_alignments = valid_symbols
			_specific_characters = '[]§«»ÀÂÇÉÊÎÔàâæçèéêëîïôùûü¬~"' # GB: new symbols for turntaking & ldots, [] are for notes, " for new terms.
	elif hps['language']=='english':
			valid_symbols = [
				'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AW0', 'AW1', 'AW2', 'AX0', 'AY0', 'AY1', 'AY2', 'B',
				'CH', 'D', 'DH', 'EA0', 'EA1', 'EA2', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
				'IA0', 'IA1', 'IA2', 'IH0', 'IH1', 'IH2', 'II0', 'II1', 'II2', 'IY0', 'JH', 'K', 'KV', 'L', 'M', 'N', 'NG', 'OH0',
				'OH1', 'OH2', 'OO0', 'OO1', 'OO2', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UA0',
				'UA1', 'UA2', 'UH0', 'UH1', 'UH2', 'UU0', 'UU1', 'UU2', 'UW0', 'V', 'W', 'Y', 'Z', 'ZH', '_', '__'
			]
			valid_alignments = valid_symbols + [
				'AA2&R', 'AX0&L', 'AX0&M', 'AX0&N', 'AX0&R', 'AY1&AX0', 'AY2&AX0', 'D&AX0', 'EA1&R', 'EH0&M', 'G&AX0', 'G&Z', 'G&ZH',
				'IH0&Z', 'K&S', 'K&SH', 'M&AE1', 'M&AX0', 'N&Y', 'T&S', 'W&AH0', 'W&AH1', 'W&AH2', 'W&OH1', 'W&OH2', 'Y&AX0', 'Y&EH0',
				'Y&EH1', 'Y&ER1', 'Y&IY0', 'Y&OO1', 'Y&UA0', 'Y&UA1', 'Y&UA2', 'Y&UH0', 'Y&UH1', 'Y&UH2', 'Y&UU0', 'Y&UU1', 'Y&UU2', 'Y&UW0'
			]
			_specific_characters = []
			
	_tokens = '01' # ML: Start of Sequence <SoS> and and of Sequence <EoS> tokens
	_pad        = '_'
	_punctuation = '!\'(),.:;? '
	_special = '-'
	_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

	# Prepend "@" to phonetic symbols to ensure uniqueness (some are the same as uppercase letters):
	_arpabet = ['@' + s for s in valid_symbols]

	# Export all symbols:
	symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_specific_characters) + _arpabet + list(_tokens) + ['#'] # GB: mark for emphasis 
	out_symbols = valid_alignments

	_symbol_to_id = {s: i for i, s in enumerate(symbols)}
	_id_to_symbol = {i: s for i, s in enumerate(symbols)}
	# Mappings from out_symbol to numeric ID and vice versa:
	_out_symbol_to_id = {s: i for i, s in enumerate(out_symbols)}
	_id_to_out_symbol = {i: s for i, s in enumerate(out_symbols)}

def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')
  
def text_to_sequence(text):
	ind=[(m.start(0), m.end(0)) for m in re.finditer('\{([^\}]+?)\}',text)] # check for curly
	if len(ind):
		deb=0; pho=[]
		for i,v in enumerate(ind):
			pho += [*text[deb:ind[i][0]]]
			pho += ['@'+m for m in txt[v[0]+1:v[1]-1].split()]
			deb = v[1]
		pho += [*text[v[1]:]]
	else:
		pho = [*text]


