import csv
import sys
import datetime
from SpellCorrector import Sentence_Corrector

if __name__ == '__main__':

    time1 = datetime.datetime.now()
    raw_file = sys.argv[1]

    # train model on the 'big.txt' file
    corrector = Sentence_Corrector('./big.txt')

    with open(raw_file, 'r', encoding='utf-8-sig') as input_file:
        reader = csv.reader(input_file)

        with open('model_output.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file)

            for row in reader:
                raw_search_term = str(row[0])
                if raw_search_term == "raw_query":
                    writer.writerow(["raw_query", "corrected_query"])
                else:
                    corrected_search_term = corrector.return_best_sentence(raw_search_term)
                    writer.writerow([raw_search_term, corrected_search_term[0]])

    time2 = datetime.datetime.now()
    # if needed, print out the time used
    print("Completed saving result as model_evaluation.csv in", (time2 - time1).total_seconds(), "secs")

