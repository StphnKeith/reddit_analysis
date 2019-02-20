import numpy as np
import sys
import argparse
import os
import json
import re
import csv

''' Created global variables for csv lists to save time.
    Used identifiers based on student number to avoid collision.
'''
bngl_1001504758 = []
with open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        bngl_1001504758.append(row)

rw_1001504758 = []
with open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        rw_1001504758.append(row)

def count_regex( comment, regex ):
    return len(re.findall(regex, comment))

def count_text( comment, words, tags ):
    ''' This function counts the number of tokens that have the given words
    as their lemmas or tags.

    If tags is set to true it counts the number of tags that match the given
    words list, otherwise it counts the lemmas.

    Parameters:
        comment : string, the comment to count from
        words   : List[string], the lemmas/tags to be counted
        tags    : bool, true if counting tags, false if counting lemmas

    Returns:
        count   : the number of occurences of the lemmas/tags in the given
                  words list
                  reg = re.compile(word + "\/\S+")
    '''
    count = 0
    if tags:
        for tag in words:
            reg = re.compile("\S+\/" + tag + " ")
            count += len(re.findall(reg, comment))
    else:
        for lemma in words:
            reg = re.compile(" " + lemma + "\/\S+")
            count += len(re.findall(reg, comment))

    return count

def search_csv( word, cols, bngl ):
    ''' This function retrieves the data from the csv files for a given word
    and some given columns.

    It will return a list of integers to be added to the feature vector.

    Parameters:
        word : string, the word to be found in the csv file
        cols : List[int], the columns of interest
        bngl : bool, true if the csv file is BristolNorms+GilhoolyLogie,
               false if it's Ratings_Warriner_et_al

    Returns:
        vals : List[int], the list of values extracted from 
    '''
    vals = len(cols) * [0]
    if bngl:
        rows = bngl_1001504758
    else:
        rows = rw_1001504758

    # Search rows
    start = 0
    end = len(rows)
    i = (end - start) // 2 # mid
    while end - start > 2:
        if rows[i][1] == word: # found
            for j in range(0,len(cols)):
                vals[j] = np.float64(rows[i][cols[j]])
            return vals
        else: # continue searching
            if rows[i][1] > word:
                end = i
            else:
                start = i
            i = ( (end - start) // 2 ) + start

    return vals # if it reaches here it's not in the csv

def csv_feats( comment, bngl ):
    ''' This function extracts the average and standard deviation of the
    columns of interest in the csv files.

    Parameters:
        comment : string, the comment to be examining
        bngl    : bool, true if the csv file is BristolNorms+GilhoolyLogie,
                  false if it's Ratings_Warriner_et_al

    Returns:
        vals    : Tuple[List[int]], the first item is the list of averages,
                  the second item is the list of standard deviations
    '''
    # Initialization:
    if bngl:
        cols = [3, 4, 5]
    else:
        cols = [2, 5, 8]

    tokens = re.sub(r"(\S+)\/\S+", r"\1", comment).split()
    token_feats = []

    for token in tokens:
        # If it's not a punctuation/symbol
        if re.match(r"[a-zA-Z]", token):
            token_feats.append(search_csv(token, cols, bngl))

    averages = np.average(token_feats, axis=0)
    deviations = np.std(token_feats, axis=0)

    return (averages, deviations)

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros( (174) )

    # 1. Number of first-person pronouns
    fpp = ["I", "me", "my", "mine", "we", "us", "our", "ours"]
    feats[0] = count_text(comment, fpp, False)

    # 2. Number of second-person pronouns
    spp = ["you", "your", "yours", "u", "ur", "urs"]
    feats[1] = count_text(comment, spp, False)

    # 3. Number of third-person pronouns
    tpp = ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"]
    feats[2] = count_text(comment, tpp, False)

    # 4. Number of coordinating conjunctions
    cc = ["cc"]
    feats[3] = count_text(comment, cc, True)

    # 5. Number of past-tense verbs
    ptv = ["vbd"]
    feats[4] = count_text(comment, ptv, True)

    # 6. Number of future-tense verbs
    ftv = ["'ll", "will", "gonna"]
    ftv_count = count_text(comment, ftv, True)
    ftv_count += count_regex(comment, r"going\/\S+ to\/\S+ \S+\/vb")
    feats[5] = ftv_count

    # 7. Number of commas
    c = [","] #tag
    feats[6] = count_text(comment, c, True)

    # 8. Number of multi-character punctuation tokens
    mcp_count = count_regex(comment, r"\.\.\.\/\S+")
    mcp_count += count_regex(comment, r"w")
    feats[7] = mcp_count

    # 9. Number of common nouns
    cn = ["nn", "nns"]
    feats[8] = count_text(comment, cn, True)

    # 10. Number of proper nouns
    pn = ["nnp", "nnps"]
    feats[9] = count_text(comment, pn, True)

    # 11. Number of adverbs
    av = ["rb", "rbr", "rbs"]
    feats[10] = count_text(comment, av, True)

    # 12. Number of wh- words
    whw = ["wdt", "wp", "wp\$", "wrb"]
    feats[11] = count_text(comment, whw, True)

    # 13. Number of slang acronyms
    sa = ["smh", "fwb", "lmfao", "lmao", "lms", "tbh", "rofl", "wtf", "bff", "wyd", "lylc", "brb", "atm", "imao", "sml", "btw", "bw", "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr", "thx", "kk", "omg", "omfg", "ttys", "afn", "bbs", "cya", "ez", "f2f", "gtr", "ic", "jk", "k", "ly", "ya", "nm", "np", "plz", "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol", "fml"]
    feats[12] = count_text(comment, sa, False)

    # 14. Number of words in uppercase (≥ 3 letters long)
    wiu_count = count_regex(comment, r"([A-Z]){3,}\/\S+")
    feats[13] = wiu_count

    # 15. Average length of sentences, in tokens # 12
    sentences = comment.rstrip().split("\n")
    s_lengths = []
    for sentence in sentences:
        s_lengths.append(len(sentence.split()))


    avg_s_length = np.average(s_lengths) if len(s_lengths) > 0 else 0
    feats[14] = avg_s_length

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    tokens = re.sub(r"(\S+)\/\S+", r"\1", comment).split()
    t_lengths = []
    for token in tokens:
        # If it's not a punctuation/symbol
        if re.match(r".*[a-zA-Z]", token):
            t_lengths.append(len(token))

    avg_t_length = np.average(t_lengths) if len(t_lengths) > 0 else 0
    feats[15] = avg_t_length

    # 17. Number of sentences.
    feats[16] = comment.count("\n")

    # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    bngl_avgs_devs = csv_feats(comment, True)
    feats[17:20] = bngl_avgs_devs[0]
    feats[20:23] = bngl_avgs_devs[1]

    # 24. Average of V.Mean.Sum from Warringer norms
    # 25. Average of A.Mean.Sum from Warringer norms
    # 26. Average of D.Mean.Sum from Warringer norms
    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    warr_avgs_devs = csv_feats(comment, False)
    feats[23:26] = warr_avgs_devs[0]
    feats[26:29] = warr_avgs_devs[1]

    return feats

def load_liwc( cat ):
    '''
    '''
    # Initialize
    feats_file = "/u/cs401/A1/feats/" + cat + "_feats.dat.npy"
    ids_file = "/u/cs401/A1/feats/" + cat + "_IDs.txt"
    data = np.load(feats_file)
    feats = {}

    i = 0
    with open(ids_file) as file:
        for line in file:
            feats[line.rstrip()] = data[i]
            i += 1

    return feats

def main( args ):

    data = json.load(open(args.input))
    length = len(data)
    feats = np.zeros( (length, 173+1) )

    liwc = {}
    liwc["Alt"] = load_liwc("Alt")
    liwc["Right"] = load_liwc("Right")
    liwc["Center"] = load_liwc("Center")
    liwc["Left"] = load_liwc("Left")

    y_dict = {'Left': 0, 'Center': 1, 'Right': 2, 'Alt': 3}

    for i in range(0,length):
        datum = data[i]
        datum_feats = extract1(datum['body'])

        # 30-173. LIWC/Receptiviti features
        datum_feats[29:173] = liwc[ datum['cat'] ][ datum['id'] ]
        datum_feats[173] = y_dict[ datum['cat'] ]
        feats[i] = datum_feats

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

