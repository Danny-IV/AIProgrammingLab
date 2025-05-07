def FindLongestWord(file_name):
    """Return the longest word in file_name"""
    # Open file
    file = open(file_name, "r")
    text = file.read()

    # Split text by whitespace
    words = text.split()
    print(words)

    # Finding the longest word
    longest_word = ""
    for word in words:
        if len(word) > len(longest_word):
            longest_word = word

    return longest_word.lower()
