import sys

# Predpokladajme, že prvý argument je text, ktorý sa má zapísať do súboru.
text_to_write = sys.argv[1] if len(sys.argv) > 1 else ""

# Open a file named 'testtext.txt' in write mode and write the text to it.
with open('testtext.txt', 'w') as file:
    file.write(text_to_write)