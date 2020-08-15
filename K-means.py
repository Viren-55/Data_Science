##BT3041 - BE15B021
##Neeraj Kumar Milan

##f = open('fullmoon.txt')
f = open('outlier.txt')
f = f.read()
f = f.split('\n')
X = []
for i in range(len(f)-1):
        p = f[i].split(',')
        X.append((float(p[0]),float(p[1])))
        
    
import random
import math
import statistics
C = []
k = int(input('Enter value of K = '))
for i in range(k):
    y = random.SystemRandom()
    centroid = y.choice(X)
    C.append(centroid)

for j in range(50):
    Sx = [[]for i in range(k)]
    Sy = [[]for i in range(k)]
    for i in range(len(X)):
        x = X[i]
        dist = []
        for j in range(len(C)):
            s = C[j]
            d = math.sqrt((x[0]-s[0])**2 + (x[1]-s[1])**2)
            dist.append(d)
            if j == k-1:
                y = dist.index(min(dist))
                Sx[y].append(x[0])
                Sy[y].append(x[1])
    Nx = []
    Ny = []
    C = []
    for i in range(k):
        Nx.append(statistics.mean(Sx[i]))
        Ny.append(statistics.mean(Sy[i]))
        C.append((statistics.mean(Sx[i]),statistics.mean(Sy[i])))

for i in range(len(Nx)):
        print('cordinates of centroid = ',(Nx[i],Ny[i]))
se = 0
for i in range(len(Nx)):
        print(len(Sx[i]))
        for j in range(len(Sx[i])):
                
                d =math.sqrt((Nx[i]-Sx[i][j])**2 + (Ny[i]-Sy[i][j])**2)
                se+=d
print('SSE = ',se)
                



import matplotlib.pyplot as plt
for i in range(k):
    
    if i ==0:
     m = 'bo'
     plt.plot(Sx[i],Sy[i], m)
    if i ==1:
     m = 'ro'
     plt.plot(Sx[i],Sy[i], m)
    if i ==2:
     m = 'go'
     plt.plot(Sx[i],Sy[i], m)
    if i ==3:
     m = 'mo'
     plt.plot(Sx[i],Sy[i], m)
    

plt.plot(Nx,Ny,'k*')
## for fulmoon
if k ==2:
    plt.title('Application of K-means clustering on fullmoon with k =2')
if k ==3:
    plt.title('Application of K-means clustering on fullmoon with k =3')

## For outlier
##if k ==2:
##    plt.title('Application of K-means clustering on fullmoon with k =2')
##if k ==4:
##    plt.title('Application of K-means clustering on fullmoon with k =4')
##
plt.show()








