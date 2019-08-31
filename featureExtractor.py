import re


class extractor:
    def __init__(self):
        pass

    def isRetweet(self, line):
        if re.search(r"[R|r][T|t]:?\s@handle:?\s?", line):
            return 1.
        return 0.

    def mentionWordRatio(self, line):
        l = re.findall(r"(?<![R|r][T|t]\s)@handle", line)
        w = self.totalWord(line)
        return l.__len__() / w

    def totalWord(self, line):
        return line.split(" ").__len__()

    def numOfURL(self, line):
        l = re.findall("http", line)
        return l.__len__()

    def hashtagWordRatio(self, line):
        l = re.findall(r"(?<!&)#", line)
        w = self.totalWord(line)
        return l.__len__() / w

    def numOfMoney(self, line):
        l = re.findall(r"\$[1-9][0-9]*", line)
        return l.__len__()

    def lineToVector(self, line):
        return [
            self.isRetweet(line),
            self.mentionWordRatio(line),
            self.totalWord(line),
            self.hashtagWordRatio(line),
            self.numOfMoney(line)
        ]

    def str(self, vec):
        s = ""
        for x in vec:
            s += str(x) + ','
        s = s[:-1]
        return s

    def batchToVector(self, d, usr_flag, save=False, file_name="data"):
        print("convert raw data to vector.....")
        if save:
            print("saving enabled, writing to file: " + file_name)
            save_file = open(file_name, 'w', encoding='utf-8')
            if usr_flag:
                for k in d.keys():
                    for l in d[k]:
                        save_file.write(k + '\t' + self.str(self.lineToVector(l)) + '\n')
            else:
                for l in d:
                    save_file.write(self.str(self.lineToVector(l)) + '\n')

            save_file.close()
            print("saving complete!")

        else:
            if usr_flag:
                new_d = d
                for k in d.keys():
                    for l in d[k]:
                        new_d[k] = self.lineToVector(l)

                return new_d

            else:
                d_arr = []
                for l in d:
                    d_arr.append(self.lineToVector(l))

                return d_arr
