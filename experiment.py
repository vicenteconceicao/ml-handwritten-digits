import knn
import digits
import sys
import os
import csv

# Cria diretório se não existir
def createDirectory(path):
        if not os.path.isdir(path):
                os.mkdir(path)

# Carrega imagens e padroniza de acordo com as variaveis X e Y
def loadImages(X, Y):
        foutImage = open("features.txt","w")
        digits.load_images('digits/data', foutImage, X, Y)
        foutImage.close

# Realiza o treinamento e teste do algorítmo KNN
def main(max, classifyMetric):
        for k in range(1, max+1):
                foutModel = open(path+str(k)+"_"+X+"_"+Y+"_"+classifyMetric+"_result_2.txt","w")
                knn.main(sys.argv[1], k, foutModel, classifyMetric)
                foutModel.close


if __name__ == "__main__":
        if len(sys.argv) != 2:
                sys.exit("Use: knn.py <data>")

        # Carrega arquivo que contém combinações de dimensões de imagens
        scalesFile = open("scales/scales.csv","r")

        with scalesFile as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                for row in rows:
                        X = row[0]
                        Y = row[1]
                        
                        # Pastas onde serão armazenados os resultados dos treinamentos/testes
                        path = "results_2/"
                        createDirectory(path)
                        path += X+"_"+Y+"/"
                        createDirectory(path)

                
                        loadImages(int(X), int(Y))

                        main(6, 'euclidean')
                        #main(6, 'manhattan')
                        print(X+"_"+Y+' euclidean')

       
        
        

        
