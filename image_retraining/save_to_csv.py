import csv

def save_to_csv(filename, data_arr):
    f = open(filename, 'a')
    with f:
        writer =csv.writer(f)
        for row in data_arr:
            writer.writerow(row)

if __name__ == '__main__':
    data_arr = [["this is the info"],['',11,12,13,14], ['',15,16,17]]
    save_to_csv('/home/duclong002/number.csv', data_arr)