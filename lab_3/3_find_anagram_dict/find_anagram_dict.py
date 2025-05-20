def finding_anagrams_dict(words: str) -> str:
    anagram_dict = {}

    for word in words:
        letter_count = [0] * 26
        for char in word:
            letter_count[ord(char) - ord('a')] += 1

        letter_count_tuple = tuple(letter_count)

        if letter_count_tuple in anagram_dict:
            anagram_dict[letter_count_tuple].append(word)
        else:
            anagram_dict[letter_count_tuple] = [word]

    return list(anagram_dict.values())


words = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
result = finding_anagrams_dict(words)
print("Found anagrams:")
print(result)
