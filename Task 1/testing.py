from keyboard_final import SmartKeyboard, CORPUS

keyboard = SmartKeyboard(CORPUS)

def autocorrect_full_sentence(sentence):
    words = sentence.split()
    return " ".join(keyboard.autocorrector.correct_word(w) for w in words)

text = "Recieve this mesage and pleaze chekc if ur keybord autocorect fixs these speling mistaks corectly, especialy the wierd ones lik 'teh' and 'wich'"
print(autocorrect_full_sentence(text))