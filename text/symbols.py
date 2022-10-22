""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# For Simplified Chinese
_punctuation_sc = '；：，。！？-“”《》、（）…— '
# The numbers are for Pinyin tones
_numbers = '123450'

# Additional symbols
# The special character symbols are used to avoid
# conflict with the letter used in Pinyin
_others = 'ＢＰ'


# Export all symbols:
symbols_en = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

symbols_cmn = [_pad] + list(_punctuation_sc) +  list(_letters) + list(_numbers) + list(_others)
