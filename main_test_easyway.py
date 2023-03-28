import csv

from openAI_API_service import *

if __name__ == '__main__':
    csvfile_data = []
    with open(FILE_NAME, mode='r') as csvfile_toread:
        csv_reader = csv.DictReader(csvfile_toread)

        for index, row in enumerate(csv_reader, 1):
            csvfile_data.append(row)

    for i, row in enumerate(csvfile_data):
        row['rate'] = fun_GPT_response(row['review text'])

    with open(FILE_NAME[:-4]+'_analyzed.csv', mode='w', newline='') as csvfile_towrite:
        csv_writer = csv.DictWriter(csvfile_towrite, fieldnames=list(csvfile_data[0].keys()))

        csv_writer.writeheader()
        for rate in range(10, 0, -1):
            csv_writer.writerows([row for row in csvfile_data if row['rate'] == rate])
