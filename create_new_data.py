import os

for letter_file in os.listdir("multiframe_csv_data"):
    nlns = []
    nount = 0
    nln = ""

    with open("multiframe_csv_data/"+letter_file, 'r') as f:
        for i, line in enumerate(f):
            if i != 0:
                if nount == 24:
                    nlns.append(nln)
                    nln = ""
                    nount = 0
                else:
                    nln += line.strip('\n')+','
                    nount += 1
            else:
                nlns.append(line.strip('\n'))
        f.close()

    with open("multiframe_csv_data/"+letter_file, 'w') as f:
        f.write("\n".join(nlns))
        f.close()
