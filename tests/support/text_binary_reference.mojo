"""Small scalar SQL LIKE oracle, independent of dataframe kernels.

The table tracks whether each pattern prefix matches each complete text prefix.
This deliberately uses a full dynamic-programming relation rather than kernel
search/backtracking, and has literal anchors in the public contract tests.
"""


def like_reference(text: String, pattern: String) -> Bool:
    var letters = List[String]()
    var tokens = List[String]()
    for letter in text.codepoints():
        letters.append(String(letter))
    for token in pattern.codepoints():
        tokens.append(String(token))
    var width = len(letters) + 1
    var matches = List[Bool](length=(len(tokens) + 1) * width, fill=False)
    matches[0] = True
    for p in range(1, len(tokens) + 1):
        for t in range(width):
            var token = tokens[p - 1]
            if token == "%":
                matches[p * width + t] = matches[(p - 1) * width + t] or (
                    t > 0 and matches[p * width + t - 1]
                )
            elif t > 0:
                matches[p * width + t] = matches[(p - 1) * width + t - 1] and (
                    token == "_" or token == letters[t - 1]
                )
    return matches[len(tokens) * width + len(letters)]
