import sys
import argparse
import os
import json
import html
import re
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])

indir = '/u/cs401/A1/data/';

def split_tokens( comment ):
    ''' Splits a tagged function into an list of (token, tag) tuples.

    This function takes a tagged comment and returns an list of tuples
    with two elements, where the first element is the token and second
    the tag. The elements of the list are in the order that they occured
    in the tagged comment.

    Parameters:
        comment    : string, the tagged comment

    Returns:
        token_tups : list, the list of token tuples
    '''
    token_tups = []
    tokens_with_tags = comment.split()
    for token_wt in tokens_with_tags:
        tup = (re.sub(r"(\S+)\/\S+", r"\1", token_wt), re.sub(r"\S+(\/\S+)", r"\1", token_wt))
        token_tups.append(tup)
    return token_tups

def rejoin_tokens( token_tups, tags=True ):
    ''' Rejoins a list of (token, tag) tuples into a single string.

    Parameters:
        token_tups : List[(string, string)], the list of token tuples
        tags       : bool, true if you want to include tags, false if
                     only the tokens should be included

    Returns:
        comment    : string, the string token_tups represents
    '''
    newComm = ""
    for tup in token_tups:
        newComm = newComm + tup[0]
        if tags:
            newComm = newComm + tup[1]
        newComm = newComm + " "
    return newComm

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    # To speed up processing a tiny bit
    if comment == "":
        return ""

    modComm = comment
    if 1 in steps:
        #print('1')
        modComm = modComm.replace("\n", " ")
        # We're replacing instead of removing so that a string like:
        #     "... end.\nStart ..."
        # becomes "... end. Start ..." instead of "... end.Start ..."
    if 2 in steps:
        #print('2')
        modComm = html.unescape(modComm)
    if 3 in steps:
        #print('3')
        modComm = re.sub(r"http\S*", "", modComm)
        modComm = re.sub(r"www\S*", "", modComm)
    if 4 in steps:
        #print('4')

        # Tokenize single periods
        modComm = re.sub(r"(?<!.[!\"#$%&\\()*+,\.\/:;<=>?@\[\]^_`{|}~]|\.[a-zA-Z])(\.)(?![a-zA-Z]\.|[!\"#$%&\\()*+,\.\/:;<=>?@\[\]^_`{|}~])", r" \1 ", modComm)

        # De-tokenize common abbreviation periods
        abbrevs = ["Capt", "Col", "Dr", "Drs", "Fig", "Figs", "Gen", "Gov", "HON", "MR", "MRS", "Messrs", "Miss", "Mmes", "Mr", "Mrs", "Ref", "Rep", "Reps", "Sen", "fig", "figs", "vs", "Lt"]

        for abb in abbrevs:
            reg = re.compile(abb + " .")
            modComm = re.sub(reg, abb + ".", modComm)

        # Tokenize multi-period abbreviations
        modComm = re.sub(r"((?:[a-zA-Z]\.){2,})", r" \1 ", modComm)

        # Tokenize groups of two or more punctuations
        modComm = re.sub(r"([!\"#$%&\\()*+,\.\/:;<=>?@\[\]^_`{|}~]{2,})", r" \1 ", modComm)

        # Tokenize single non-period punctuation
        modComm = re.sub(r"(?<![!\"#$%&\\()*+,\.\/:;<=>?@\[\]^_`{|}~])([!\"#$%&\\()*+,\/:;<=>?@\[\]^_`{|}~])(?![!\"#$%&\\()*+,\.\/:;<=>?@\[\]^_`{|}~])", r" \1 ", modComm)

    if 5 in steps:
        # Tokenize one letter prefix clitics e.g. y'all and t'challa become y' all and t' challa
        modComm = re.sub(r" ([a-zA-Z])'", r" \1' ", modComm)
        modComm = modComm.replace("n't", " n't ")
        modComm = modComm.replace("'d", " 'd ")
        modComm = modComm.replace("'ve", " 've ")
        modComm = modComm.replace("'re", " 're ")
        modComm = modComm.replace("'ll", " 'll ")
        modComm = modComm.replace("'s", " 's ")
        modComm = modComm.replace("s'", "s ' ")
    if 6 in steps:
        #print('6')
        # Remove multiple spaces
        modComm = re.sub(r" +", r" ", modComm)

        # Tag the tokens
        utt = nlp(modComm)

        newComm = ""
        for token in utt:
            if token.tag_ != "":
                newComm = newComm + token.text + "/" + token.tag_ + " "

        modComm = newComm
    if 7 in steps:
        #print('7')
        with open("/u/cs401/Wordlists/StopWords") as file:
            for line in file:
                # Construct regex
                reg = re.compile(" " + line.rstrip() + "/\S+")
                modComm = re.sub(reg, "", modComm)

    if 8 in steps:
        #print('8')
        tokens_w_tags = split_tokens(modComm)
        tempComm = rejoin_tokens(tokens_w_tags, False)
        newComm = ""
        utt2 = nlp(tempComm)

        if (len(utt2) != len(tokens_w_tags)):
            print("odd comment")
            for token in tokens_w_tags:
                utt = nlp(token[0])
                if utt[0].lemma_[0] == "-":
                    lemma = token[0]
                else:
                    lemma = utt[0].lemma_
                newComm = newComm + lemma + token[1] + " "
        else: # This method is faster but doesn't work if len(utt2) != len(tokens_w_tags)
            i = 0
            for token in utt2:
                if token.lemma_[0] == "-":
                    lemma = tokens_w_tags[i][0]
                else:
                    lemma = token.lemma_
                newComm = newComm + lemma + tokens_w_tags[i][1] + " "
                i += 1

        modComm = newComm
    if 9 in steps:
        #print('9')
        '''Since lemmatization in step 8 makes everything lowercase,we'll just
        assume every punctuation that isn't a period preceded by a common
        abbreviation, or a punctuation at the end of a punctuation group is an
        end-of-sentence punctuation.
        '''

        tokens_w_tags = split_tokens(modComm)
        if len(tokens_w_tags) != 0:
            abbrev_tokens = ["col", "drs", "fig", "fig", "hon", "mr", "mrs", "Mississippi", "mmes", "ref", "reps", "fig", "lt"]
            newComm = " " + tokens_w_tags[0][0] + tokens_w_tags[0][1] + " "

            i = 1
            while i < len(tokens_w_tags):
                token = tokens_w_tags[i][0]
                tag = tokens_w_tags[i][1]
                newComm = newComm + token + tag

                # Case 1: it's a period
                if token == "." and tokens_w_tags[i-1][0] not in abbrev_tokens and (i + 1 >= len(tokens_w_tags) or tokens_w_tags[i+1][1] != "/."):
                    # Then this is a period not preceded by a common punctuation,
                    # and is succeded by non-punctuation, and is therefore probably
                    # the end of a sentence.
                    newComm = newComm + " \n "
                    i += 1
                    continue

                # Case 2: it's not a period
                if token != "." and tag == "/." and (i + 1 >= len(tokens_w_tags) or tokens_w_tags[i+1][1] != "/."):
                    # Then this is a punctuation that is followed by a
                    # non-punctuation and is therefore probably the end of
                    # the sentence.
                    newComm = newComm + " \n "
                    i += 1
                    continue

                # Case 3: it's not punctuation at the end of a sentence
                newComm = newComm + " "
                i += 1

            modComm = newComm

    if 10 in steps:
        #print('10')
        modComm = re.sub(r"(\S+)(\/\S+)", lambda x: x.group(1).lower() + x.group(2), modComm)
        
    return modComm

def main( args ):

    allOutput = []
    count = 0
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            student_id = args.ID[0]
            max_lines = args.max
            start = student_id % len(data)
            end = start + max_lines
            length = len(data)

            # TODO: read those lines with something like `j = json.loads(line)`
            for i in range(start, end):
                line = json.loads(data[i % length])
                # TODO: choose to retain fields from those lines that are relevant to you
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # TODO: replace the 'body' field with the processed text
                selected_fields = {'id': line['id'], 'body': preproc1(line['body']), 'cat': file}
                # TODO: append the result to 'allOutput'
                allOutput.append(selected_fields)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
