import knn
import digits
import sys
import os
import csv

def createDirectory(path):
        if not os.path.isdir(path):
                os.mkdir(path)

def loadImages(X, Y):
        foutImage = open("features.txt","w")
        digits.load_images('digits/data', foutImage, X, Y)
        foutImage.close

def main(max, classifyMetric):
        for k in range(1, max+1):
                foutModel = open(path+str(k)+"_"+X+"_"+Y+"_"+classifyMetric+"_result.txt","w")
                knn.main(sys.argv[1], k, foutModel, classifyMetric)
                foutModel.close


if __name__ == "__main__":
        if len(sys.argv) != 2:
                sys.exit("Use: knn.py <data>")

        scalesFile = open("scales/scales.csv","r")

        with scalesFile as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                for row in rows:
                        X = row[0]
                        Y = row[1]
                        
                        path = "results/"
                        createDirectory(path)
                        path += X+"_"+Y+"/"
                        createDirectory(path)

                
                        loadImages(int(X), int(Y))

                        main(10, 'manhattan')

       
        
        

        
