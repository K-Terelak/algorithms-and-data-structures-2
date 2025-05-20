from collections import defaultdict


def finding_anagrams_sort(words):
    anagram_dict = defaultdict(list)

    for word in words:
        anagram_dict[''.join(sorted(word))].append(word)

    return list(anagram_dict.values())


words = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
result = finding_anagrams_sort(words)
print("Found anagrams:")
print(result)
