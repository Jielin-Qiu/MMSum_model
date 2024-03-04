#!/usr/bin/env python3
#
# This file is part of SumeCzech corpus <http://hdl.handle.net/11234/1-2615>.
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import re

class RougeRaw:
    """Compute RougeRAW-1, RougeRAW-2, RougeRAW-L metrics."""

    class FScore:
        """F1 score representation."""
        def __init__(self, correct, gold, system):
            self.p = correct / system if system else 0.
            self.r = correct / gold if gold else 0.
            self.f = 2 * correct / (system + gold) if system + gold else 0.

    def _rouge_n(self, n, gold_words, system_words):
        """Compute Rouge-n for given words."""
        def n_grams(n, words):
            ngrams = {}
            total = 0
            for i in range(len(words) - n + 1):
                ngram = "\t".join(words[i:i + n])
                ngrams[ngram] = 1 + ngrams.get(ngram, 0)
                total += 1
            return ngrams, total

        gold_ngrams, gold_total = n_grams(n, gold_words)
        system_ngrams, system_total = n_grams(n, system_words)

        intersection = 0
        for ngram in system_ngrams:
            intersection += min(system_ngrams[ngram], gold_ngrams.get(ngram, 0))

        return self.FScore(intersection, gold_total, system_total)

    def _rouge_l(self, gold_words, system_words):
        """Compute Rouge-L for given words."""
        lcs = [[0] * len(system_words) for _ in gold_words]
        for r in range(len(gold_words)):
            for s in range(len(system_words)):
                if gold_words[r] == system_words[s]:
                    lcs[r][s] = 1 + (lcs[r - 1][s - 1] if r and s else 0)
                lcs[r][s] = max(lcs[r][s], lcs[r - 1][s] if r else 0)
                lcs[r][s] = max(lcs[r][s], lcs[r][s - 1] if s else 0)

        return self.FScore(lcs[-1][-1], len(gold_words), len(system_words))

    def _tokenize(self, text):
        """Tokenize given text."""
        return re.sub(r"\s+", " ", re.sub(r"\b", " ", text, re.UNICODE), re.UNICODE).strip().split(" ")

    def document(self, gold, system):
        """Compute RougeRAW-1, RougeRAW-2, RougeRAW-L for given documents.

        Each document should be a string.
        """

        assert isinstance(gold, str) and isinstance(system, str), "Expected string arguments"

        lc_gold_words = [word.lower() for word in self._tokenize(gold)]
        lc_system_words = [word.lower() for word in self._tokenize(system)]

        return {
            "1": self._rouge_n(1, lc_gold_words, lc_system_words),
            "2": self._rouge_n(2, lc_gold_words, lc_system_words),
            "L": self._rouge_l(lc_gold_words, lc_system_words),
        }

    def corpus(self, gold, system):
        """Compute RougeRAW-1, RougeRAW-2, RougeRAW-L for given corpora.

        Each corpus should be a collection of documents, each document a string.
        """

        assert isinstance(gold, list) and isinstance(system, list), "Expected list arguments"
        assert len(gold) == len(system), "Given corpora should be of the same length"

        rouge = {key: self.FScore(0, 0, 0) for key in ["1", "2", "L"]}

        if len(gold):
            for gold_document, system_document in zip(gold, system):
                for key, value in self.document(gold_document, system_document).items():
                    rouge[key].p += value.p
                    rouge[key].r += value.r
                    rouge[key].f += value.f

            for key in rouge:
                rouge[key].p /= len(gold)
                rouge[key].r /= len(gold)
                rouge[key].f /= len(gold)

        return rouge


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=str, help="Gold jsonl file path")
    parser.add_argument("system", type=str, help="System jsonl output file")
    parser.add_argument("field", type=str, help="Which jsonl field to compare")
    args = parser.parse_args()

    gold = []
    with open(args.gold, "r", encoding="utf-8") as gold_file:
        for gold_line in gold_file:
            gold.append(json.loads(gold_line)[args.field])

    system = []
    with open(args.system, "r", encoding="utf-8") as system_file:
        for system_line in system_file:
            system.append(json.loads(system_line)[args.field])

    rouge = RougeRaw().corpus(gold, system)
    print("  RougeRAW-1      RougeRAW-2      RougeRAW-L")
    print("  P    R    F     P    R    F     P    R    F")
    for metric in ["1", "2", "L"]:
        print("{:04.1f} {:04.1f} {:04.1f}{}".format(
            100 * rouge[metric].p,
            100 * rouge[metric].r,
            100 * rouge[metric].f,
            "\n" if metric == "L" else "  "), end="")
