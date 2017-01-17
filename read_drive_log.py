import csv

with open('driving_log.csv', 'r') as csvfile:
  reader = csv.DictReader(csvfile, fieldnames=['front', 'left', 'right', 'steering', 'unknown1', 'unknown2', 'speed'])#, delimiter=',', quotechar='|')
  for row in reader:
    print(row['front'])
