def find_words_with_max_char(sentence, search_char):
    if search_char.isalpha() and sentence:
        max_count = 0
        words_with_max_count = []

        for word in sentence:
            if word.isalnum():
                char_count = word.count(search_char)  
                if char_count > max_count:
                    max_count = char_count
                    words_with_max_count = [word]  
                elif char_count == max_count:
                    words_with_max_count.append(word)
        return (max_count, words_with_max_count[:1])
    else:
        return (0, [])


def main():
    sentence = input("Enter Sentence (space-separated): ").split()
    search_char = input("Enter character to find word with its first max occurance: ")
    max_count, words_with_max_count = find_words_with_max_char(sentence, search_char)
    if words_with_max_count and max_count > 0:
        print(f"Words with the most '{search_char}'s ({max_count}): {' '.join(words_with_max_count)}")
    else:
        if search_char.isalpha() and sentence:
            print("No matching character found in sentence.")
        else:
            print("No proper input provided (enter alphabetic search character and at least one alphanumeric word).")

if __name__ == "__main__":
    main()
