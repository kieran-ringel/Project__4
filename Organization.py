import pandas as pd
import random
import math

class Org():
    def __init__(self, file_name, header, class_loc, discrete):
        self.file_name = file_name
        self.header = header
        self.class_loc = class_loc
        self.discrete = discrete

    """Kieran Ringel
    Takes all files and standardized them so they are formatted the same.
    This removes the header included on any file, and
    moves the class to the last column.
    Machines removes the ERP(estimated relative performance from the original article) column.
    Glass removes the index in the first column."""
    def open(self):
        file = open(self.file_name, 'r')        #opens file
        df = pd.DataFrame([line.strip('\n').split(',') for line in file.readlines()])   #splits file by lines and commas

        if self.header != [-1]:             #if user input that the data includes a header
            df = df.drop(self.header, axis=0)   #drop the header
            df = df.reset_index(drop=True)      #reset the axis
        if self.class_loc != -1:            #if the class is not in the last row
            end = df.shape[1] - 1           #moves class to last column
            col = df.pop(0)
            df.insert(end, end + 1, col)
            df.columns = df.columns - 1

        if self.file_name == "Data/glass.data" or self.file_name =="Data/breast-cancer-wisconsin.data": #if file is glass or breast cancer data
            df = df.drop(0, axis=1)             #remove column of index
            df = df.reset_index(drop=True)      #reset axis

        elif self.file_name == "Data/machine.data": #if file is machine data
            df = df.drop(0, axis=1)                 #remove vendor name
            df = df.drop(1, axis=1)                 #remove model name CHECK
            df = df.drop(9, axis=1)                 #remove column with ERP
            df = df.reset_index(drop=True)          #reset acis

        df.columns = range(df.shape[1])
        df.columns = [*df.columns[:-1], "class"]    #give column containing class label 'class'
        newdf = self.missingData(df)
        df = self.normalize(newdf)
        return(df)      #returns edited file

    def missingData(self, df):      #CHANGEEEE
        print(df)
        for column in range(df.shape[1] - 1):
            for row in range(df.shape[0]):
                if df[column][row] == '?':
                    dataclass = df['class'][row]
                    newdf = 
                    mode = df[column].mode()
                    print(mode[0])
                    #df[column][row] = random.choice(df[column])
        return(df)

    def normalize(self, file):
        """Kieran Ringel
        Normalizes all real valued data points using z score normalization"""
        for column in file.iloc[:,:-1]:
            mean= 0
            sd = 0
            if column not in self.discrete:
                for index,row in file.iterrows():
                    mean += float(file[column][index])
                mean /= file.shape[0]                   #calcualates the mean value for each attribute
                for index,row in file.iterrows():
                    sd += (float(file[column][index]) - mean) ** 2
                sd /= file.shape[0]
                sd = math.sqrt(sd)                      #calculated the standard deviation for each attribute
                for index, row in file.iterrows():
                    if sd == 0:
                        file[column][index] = mean      #gets rid of issue of sd = 0
                    else:
                        file[column][index] = (float(file[column][index]) - mean) / sd  #changed value in file to standardized value
        return(file)
