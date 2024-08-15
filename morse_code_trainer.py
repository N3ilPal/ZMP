import random

# Morse code dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----'
}

# Generate a random letter
def get_random_letter():
    return random.choice(list(MORSE_CODE_DICT.keys()))

def morse_code_trainer():
    while True:
        random_letter = get_random_letter()
        print(f"Translate the following letter to Morse code: {random_letter}")
        
        user_input = input("Enter the Morse code (use . for dots and - for dashes), or type 'exit' to quit: ").strip()

        if user_input.lower() == "exit":
            print("Exiting the Morse code trainer.")
            break

        if user_input == MORSE_CODE_DICT[random_letter]:
            print("Correct!")
        else:
            print(f"Incorrect. The correct Morse code for {random_letter} is {MORSE_CODE_DICT[random_letter]}")
            print(" ")

if __name__ == "__main__":
    morse_code_trainer()
