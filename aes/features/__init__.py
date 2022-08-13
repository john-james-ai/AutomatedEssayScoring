#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /__init__.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 2nd 2022 08:08:36 pm                                                 #
# Modified   : Thursday August 11th 2022 02:37:02 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
FEATURES = {
    "length": [
        "alphabetic_character_count",
        "number_character_count",
        "special_character_count",
        "word_count",
        "word_count_gt_5",
        "word_count_gt_6",
        "word_count_gt_7",
        "word_count_gt_8",
        "vocabulary_size",
        "sentence_count",
        "lemmas_count",
        "punctuation_count",
        "commas_count",
        "exclamation_mark_count",
        "question_mark_count",
        "avg_word_length",
        "std_word_length",
        "avg_sentence_length",
        "std_sentence_length",
    ],
    "syntactic": [
        "noun_count",
        "verb_count",
        "adjective_count",
        "adverb_count",
        "conjunction_count",
        "type_token_ratio",
        "existential_there_count",
        "superlative_count",
        "verb_compliment_counts",
        "noun_complement_counts",
        "adjective_complement_counts",
        "that_relative_clause_count",
        "wh_relative_clause_count",
        "pre_quallifier_count",
        "pre_quantifier_count",
        "post_determiner_count",
        "demonstrative_determiner_count",
        "singular_article_count",
        "definite_article_count",
        "indefinite_article_count",
        "singular_determiner_count",
        "plural_determiner_count",
        "double_conjunction_count",
        "attributive_adjective_count",
        "post_noun_modifying_prepositional_phrase",
    ],
    "word": [
        "spelling_error_count",
        "spelling_error_ratio",
        "stop_word_count",
        "stop_word_ratio",
        "bigram_count",
        "trigram_count",
        "stemmed_bigram_count",
        "stemmed_trigram_count",
        "modal_count",
    ],
    "readability": [
        "automated_readibility_index",
        "Coleman-Liau index",
        "Dale-Chall readability score",
        "Difficult word count",
        "Flesch reading ease",
        "Flesch-Kincaid grade",
        "Gunning fog",
        "Linsear write formula",
        "Smog index",
        "Syllables count",
    ],
    "semantic": ["similarity", "histogram_based"],
}
