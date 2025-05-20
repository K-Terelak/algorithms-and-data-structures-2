def finding_palindromes(s: str) -> str:
    n, ans = len(s), []

    def addPalindrome(left: int, right: int) -> int:
        temp = []
        while left >= 0 and right < n and s[left] == s[right]:
            temp.append(s[left:right + 1])
            left -= 1
            right += 1
        return temp

    for i in range(n):
        even = addPalindrome(i, i + 1)
        odd = addPalindrome(i, i)
        ans += even + odd

    return ans


s = "ccbbdddbddb"
result = finding_palindromes(s)
print("Found palindromes:")
print(result)
