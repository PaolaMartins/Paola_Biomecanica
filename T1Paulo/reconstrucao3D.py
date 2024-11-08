# -*- coding: utf-8 -*-

#  Para compilar, deve acessar o caminho da pasta onde está no terminal, então é só colocar o seguinte comando:
# "python3 reconstrucao3D.py c1.txt c2.txt c1cal.txt c2cal.txt calibrador_ref.txt test"


# Bibliotecas utilizadas
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy.linalg import inv
from numpy.linalg import pinv
import sys


# %% Calibration DLT
def dlt_calib(cp3d, cp2d):
    pontos_3d_ref= np.asarray(cp3d)
    pontos_2d_ref= np.asarray(cp2d)

    m = np.size(cp3d[:, 0], 0)
    M = np.zeros([m * 2, 11])
    N = np.zeros([m * 2, 1])

    for i in range(m):
        M[i*2,:] = [cp3d[i,0], cp3d[i,1], cp3d[i,2] ,1, 0, 0, 0, 0, -cp2d[i, 0] * cp3d[i, 0], -cp2d[i, 0] * cp3d[i, 1], -cp2d[i, 0] * cp3d[i, 2]]
        M[i*2+1,:] = [0 , 0, 0, 0, cp3d[i,0], cp3d[i,1], cp3d[i,2],1, -cp2d[i,1] * cp3d[i,0],-cp2d[i,1] * cp3d[i,1], -cp2d[i,1] * cp3d[i,2]]
        N[[i*2,i*2+1],0] = cp2d[i, :]

    Mt = M.T
    M1 = inv(Mt.dot(M))
    MN = Mt.dot(N)

    DLT = (M1).dot(MN).T

    return DLT

# %% Reconstruction 3D
def r3d(DLTs, pontos_2d):
    DLTs = np.asarray(DLTs)
    pontos_2d = np.asarray(pontos_2d)
    
    m = len(DLTs)
    M = np.zeros([2 * m, 3])
    N = np.zeros([2 * m, 1])

    for i in range(m):
        M[i*2,:] = [DLTs[i,0]-DLTs[i,8]*pontos_2d[i,0], DLTs[i,1]-DLTs[i,9]*pontos_2d[i,0], DLTs[i,2]-DLTs[i,10]*pontos_2d[i,0]]
        M[i*2+1,:] = [DLTs[i,4]-DLTs[i,8]*pontos_2d[i,1],DLTs[i,5]-DLTs[i,9]*pontos_2d[i,1],DLTs[i,6]-DLTs[i,10]*pontos_2d[i,1]]
        Np1 = pontos_2d[i,:].T
        Np2 = [DLTs[i,3],DLTs[i,7]]
        N[[i*2,i*2+1],0] = Np1 - Np2

    coord_3d = inv(M.T.dot(M)).dot((M.T.dot(N)))
    
    return coord_3d

# Run in IDE Python
def rec3d_ide(c1=None, c2=None, ref=None):
    # Para teste na IDE
    if c1 is None:
        dfcp2d_c1 = pd.read_csv('cp2d_c1.txt', delimiter=' ',header=None)
        dfcp2d_c2 = pd.read_csv('cp2d_c2.txt', delimiter=' ',header=None)
        dfcp3d = pd.read_csv('cp3d.txt', delimiter=' ',header=None)
    else:
        dfcp2d_c1 = c1
        dfcp2d_c2 = c2
        dfpontos_3d_ref= ref
            
    pontos_2d_camera1 = np.asarray(dfcp2d_c1)
    pontos_2d_camera2 = np.asarray(dfcp2d_c2)
    pontos_3d_ref= np.asarray(dfcp3d)
    
    dlt_params_c1 = dlt_calib(cp3d, pontos_2d_camera1)
    dlt_params_c2 = dlt_calib(cp3d, pontos_2d_camera2)
    
    DLTs = np.append(dlt_params_c1, dlt_params_c2, axis=0)
    
    coord_3d = np.zeros([len(pontos_2d_camera1), 3])
    for i in range(len(pontos_2d_camera1)):
        pontos_2d = np.append([pontos_2d_camera1[i, :]], [pontos_2d_camera2[i, :]], axis=0)
        coord_3d[i, :] = r3d(DLTs, pontos_2d).T
    
    return coord_3d


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


# Run CMD or Terminal Shell
if __name__ == '__main__':
    print('\n')

    # Definindo os parâmetros das cameras do experimento
    resolutionx = int(720/2) # Arrumar a translação X do sist. ref do kinovea 0.9.4
    resolutiony = int(220/2) # Arrumar a translação Y do sist. ref do kinovea 0.9.4
    freq = 120 # Frequência de amostragem
    
    # Carregando os arquivos das cameras do Kinovea
    # Camera 1
    bola1 = pd.read_csv(str(sys.argv[1]), sep=r'\s+', header=None, decimal='.')
    bola1[1] = bola1[1] - -resolutionx
    bola1[2] = -1 * (bola1[2] - resolutiony) 
    bola1 = np.asarray(bola1[[1,2]])
    bola1b = bola1
    
    # Camera 2
    bola2 = pd.read_csv(str(sys.argv[2]), sep=r'\s+', header=None, decimal='.')
    bola2[1] = bola2[1] - -resolutionx
    bola2[2] = -1 * (bola2[2] - resolutiony) 
    bola2 = np.asarray(bola2[[1,2]])
    bola2b = bola2
    
    idx = np.asarray(list(range(len(bola1b))))
    diffball = abs(np.diff(bola1b[:,0])) > 5
    diffball = np.insert(diffball, 0, False)
    phitball = idx[diffball][0]
    idxbefore = idx[0:phitball-2]
    idxafter = idx[phitball+1::]
    idxcimpact = idx[phitball-2:phitball+1]

    print(f'Frame of impact = {phitball}')
    print(f'Critical impact frames  = {idxcimpact}')

    plt.close('all')
    plt.subplot(2,1,1)
    plt.grid(True)
    plt.plot(bola1[:,0],bola1[:,1],'o')
    plt.xlabel('CAM 1 - Coordenada X')
    plt.ylabel('CAM 1 - Coordenada Y')
    resx = 2 * resolutionx
    resy = 2 * resolutiony
    plt.title(f'Pixels coordenadas (resolução = {resx} X {resy})')
    
    plt.subplot(2,1,2)
    plt.plot(bola2[:,0],bola2[:,1],'o')
    plt.xlabel('CAM 2 - Coordenada X')
    plt.ylabel('CAM 2 - Coordenada Y')
    plt.grid(True)
    
    # Carregar arquivos de calibracao 
    datcal_c1 = np.asarray(pd.read_csv(str(sys.argv[3]), sep=r'\s+', header=None))
    datcal_c1[:, 0] = datcal_c1[:, 0] - -resolutionx
    datcal_c1[:, 1] = -1 * (datcal_c1[:, 1] - resolutiony) 
    
    datcal_c2 = np.asarray(pd.read_csv(str(sys.argv[4]), sep=r'\s+', header=None))
    datcal_c2[:, 0] = datcal_c2[:, 0] - -resolutionx
    datcal_c2[:, 1] = -1 * (datcal_c2[:, 1] - resolutiony) 

    
    ref = np.asarray(pd.read_csv(sys.argv[5], sep=r'\s+', header=None))
    ref = ref[:,1:]
    
    
    dlt_params_c1 = dlt_calib(ref, datcal_c1)
    dlt_params_c2 = dlt_calib(ref, datcal_c2)
    dlts = np.append(dlt_params_c1, dlt_params_c2, axis=0)
    
    coord_3d = np.zeros([len(bola1), 3])
    
    for i in range(len(bola1)):
        pontos_2d = np.append([bola1[i, :]], [bola2[i, :]], axis=0)
        coord_3d[i, :] = r3d(dlts, pontos_2d).T
    
    coord_3df = coord_3d[idxafter,:]
    coefsx = np.polyfit(idxafter, coord_3df[:,0], 1)
    coefsy = np.polyfit(idxafter, coord_3df[:,1], 1)
    coefsz = np.polyfit(idxafter, coord_3df[:,2], 2)
    
    coord_3df[:,0] = coefsx[0] * idxafter + coefsx[1]
    coord_3df[:,1] = coefsy[0] * idxafter + coefsy[1]
    coord_3df[:,2] = coefsz[0] * idxafter**2 + coefsz[1] * idxafter + coefsz[2]
    
    velocidades = (np.sqrt(np.sum((np.diff(coord_3df, axis=0)**2), axis=1))) / (1/freq) * 3.6
    print(f'Velocidade = {velocidades}')

    vsaida = coord_3df[-1:,:] - coord_3d[0,:]

    azimuth, elevation, r = cart2sph(vsaida[0][0], vsaida[0][1], vsaida[0][2])
    pi = np.pi
    azi = azimuth * 180/pi
    elev = elevation * 180/pi
    
    print(f'Angulos: azimuth = {azi}; elevação = {elev}')
   
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d') 
    ax2.plot3D(coord_3d[:,0], coord_3d[:,1], coord_3d[:,2], 'ro', markersize=10)
    ax2.plot3D(ref[:,0], ref[:,1], ref[:,2], 'b.')
    ax2.plot3D(coord_3df[:,0], coord_3df[:,1], coord_3df[:,2], 'k-o')
    ax2.plot3D([coord_3d[0,0],coord_3d[0,0]], [coord_3d[0,1],coord_3d[0,1]], [coord_3d[0,2],coord_3d[0,2]], 'g.', markersize=10)
    
    ax2.set_zlabel('Z [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_xlabel('X [m]')
    
    # print(coord_3df)
    distanciaVet = np.sqrt(np.sum((coord_3df[-1,:] - coord_3df[0,:])**2))
    
    veloMedia = distanciaVet / (len(coord_3df) * (1/freq)) * 3.6
    plt.title(f'Velocidade (Max = {np.round(max(velocidades),2)} km/h ; Média = {np.round(veloMedia)}); Angulos (azi = {np.round(azi,1)}, elev. = {np.round(elev,1)})')
    plt.show()

    resultado = list(np.append(velocidades, [azi, elev, veloMedia]))
    np.savetxt(str(sys.argv[6])+'_result.txt', resultado, fmt='%.10f')
    print(f'Veloc. Média = {veloMedia}')
    print('\n')

   
    np.savetxt(str(sys.argv[6])+'.3d', coord_3d, fmt='%.10f')
    np.savetxt(str(sys.argv[6])+'_filt.3d', coord_3df, fmt='%.10f')

    np.savetxt(str(sys.argv[6])+'.3d', coord_3d, fmt='%.10f')