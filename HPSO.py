 # -*- coding: utf-8 -*-
from cmath import cos, pi
import math
import random
from telnetlib import PRAGMA_HEARTBEAT
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import time as tm
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei'] 


def calDistance(CityCoordinates):
    dis_matrix = pd.DataFrame(data=None,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi,yi = CityCoordinates[i][0],CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj,yj = CityCoordinates[j][0],CityCoordinates[j][1]
            dis_matrix.iloc[i,j] = round(math.sqrt((xi-xj)**2+(yi-yj)**2),2)
    return dis_matrix


def assign_distribution_center(dis_matrix,DC,C):
    d = [[] for i in range(DC)]
    for i in range(DC,DC+C):
        d_i = [dis_matrix.loc[i,j] for j in range(DC)]
        min_dis_index = d_i.index(min(d_i))
        d[min_dis_index].append(i)   
    return d


def greedy(CityCoordinates,dis_matrix,DC,certer_number):
    dis_matrix = dis_matrix.iloc[[certer_number]+CityCoordinates,[certer_number]+CityCoordinates].astype('float64')
    for i in CityCoordinates:dis_matrix.loc[i,i]=math.pow(10,10)
    dis_matrix.loc[:,certer_number]=math.pow(10,10)
    line = []
    now_cus = random.sample(CityCoordinates,1)[0]
    line.append(now_cus)
    dis_matrix.loc[:,now_cus] = math.pow(10,10)
    for i in range(1,len(CityCoordinates)):
        next_cus = dis_matrix.loc[now_cus,:].idxmin()
        line.append(next_cus)
        dis_matrix.loc[:,next_cus] = math.pow(10,10)
        now_cus = next_cus
    return line


def calFitness(birdPop,certer_number,Demand,dis_matrix,CAPACITY,DISTABCE,C0,C1,C2,C3,time,V):
    birdPop_car,fits,distance = [],[],[]
    c0s=C0
    for j in range(len(birdPop)):
        C0=c0s
        bird = birdPop[j]
        lines = []
        line = [certer_number]
        dis_sum = 0
        dis,d = 0,0
        i = 0
        time_point = 0
        wait = 0
        late = 0
        r=0
        while i < len(bird):
            if line == [certer_number]:
                dis += dis_matrix.loc[certer_number,bird[i]]
                line.append(bird[i])
                d += Demand[bird[i]]
                time_point += dis_matrix.loc[certer_number,bird[i]]/V
                if time_point < time[bird[i]][0]:
                    wait = time[bird[i]][0] - time_point
                    time_point = time_point + wait + time[bird[i]][2]
                elif time_point > time[bird[i]][1]:
                    late = time_point - time[bird[i]][1]
                    time_point = time_point + time[bird[i]][2]
                else:
                    time_point = time_point + time[bird[i]][2]
                i += 1
            else:
                if (dis_matrix.loc[line[-1],bird[i]]+dis_matrix.loc[bird[i],certer_number]+ dis <= DISTABCE[r]) & (d + Demand[bird[i]]<=CAPACITY[r]) :

                    dis += dis_matrix.loc[line[-1],bird[i]]
                    time_point += dis_matrix.loc[line[-1],bird[i]]/V
                    if time_point < time[bird[i]][0]:
                        wait = time[bird[i]][0] - time_point
                        time_point = time_point + wait + time[bird[i]][2]
                    elif time_point > time[bird[i]][1]:
                        late = time_point - time[bird[i]][1]
                        time_point += time[bird[i]][2]
                    else:
                        time_point = time_point + time[bird[i]][2]
                    line.append(bird[i])
                    d += Demand[bird[i]]
                    r+=1
                    i += 1                         
                else:
                    dis += dis_matrix.loc[line[-1],certer_number]
                    line.append(certer_number)
                    dis_sum += dis
                    lines.append(line)
                    dis,d = 0,0
                    line = [certer_number]
                    time_point =0
                    r+=1  
        dis += dis_matrix.loc[line[-1],certer_number]
        line.append(certer_number)
        dis_sum += dis
        lines.append(line)
        distance.append(round(dis_sum,1)) 
        birdPop_car.append(lines)
      
        fits.append(round(C1*dis_sum+C0[0]*len(lines)+C2*wait+C3*late,1))
       
        look_line=lines
    return birdPop_car,fits,distance,look_line



""" def pso_algor(self, r1, r2, c_iter):
        '''
        粒子群算法公式
        Param: 
            r1 [0,1）Float 之间的伪随机数
            r2 [0,1）Float 之间的伪随机数
            c_iter  Int    当前迭代次数
        '''
        w_max = self.w[0]
        w_min = self.w[1]

        # 更新速度
        self.v = self.w * self.v + self.c1 * r1 * \
            (self.p_best - self.x) + self.c2 * r2 * \
            (self.g_best - self.x)
        # 更新例子位置
        self.x = self.x + self.v

        # 递减W，暂时未启用
        w = w_max + (self.iteration_steps - c_iter) * \   
                    (w_max - w_min) / self.iteration_steps
        self.w = w
 """


def crossover(bird,pLine,gLine,w,c1,c2,wlist,c1a,c1b,c2a,c2b,fit):
    croBird = [None]*len(bird)
    parent1 = bird
    randNum = random.uniform(0, sum([w,c1,c2]))
    w_max =max(wlist)
    w_min =min(wlist)
    """ avg_fit=np.mean(fits)
    min_fit=min(fits)
    max_fit=max(fits)
    now_fit=fits[-1]
    if now_fit <= avg_fit:
        w =min_fit - (max_fit - min_fit)*\
            (now_fit -min_fit)/(avg_fit-min_fit)
    else:
        w =max_fit """
    """ w =w_min+(w_max - w_min)*\
                ((iterI/iterMax) ** 0.5) """

    w = w_max -(iterI ** 2)*\
            (w_max - w_min) / (iterMax ** 2)   
    """ c1a=2*w_max
    c1b=w_min
    c2a=w_min
    c2b=2*w_max """
    """ c1 =c1a +(c1b - c1a)*iterI/iterMax
    c2 =c2a +(c2b - c2a)*iterI/iterMax """
    arf=2
    signm=0.5
    c1 =arf * math.sin((1-iterI/iterMax)* pi/2 )+ signm
    c2=arf * math.cos((1-iterI/iterMax)* pi/2) + signm
    if randNum <= w:
        parent2 = [bird[i] for i in range(len(bird)-1,-1,-1)]#bird的逆序
    elif randNum <= w+c1:
        parent2 = pLine
    else:
        parent2 = gLine
    start_pos = random.randint(0,len(parent1)-1)
    end_pos = random.randint(0,len(parent1)-1)
    if start_pos>end_pos:start_pos,end_pos = end_pos,start_pos
    croBird[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
    list2 = list(range(0,start_pos))
    list1 = list(range(end_pos+1,len(parent2)))
    list_index = list1+list2
    j = -1
    for i in list_index:
        for j in range(j+1,len(parent2)+1):
            if parent2[j] not in croBird:
                croBird[i] = parent2[j]
                break           
    return croBird,w,c1,c2


def mutation(croBird):
    r1 = np.random.randint(0,len(croBird)-1)
    r2 = np.random.randint(0,len(croBird)-1)
    while r2 == r1:
        r2 = np.random.randint(0,len(croBird)-1)   
    croBird[r1], croBird[r2] = croBird[r2], croBird[r1]
    return croBird,r1,r2


def draw_path(car_routes,CityCoordinates):
    shape = ['*-','*-','*-','*-','*-','*-','*-','*-','*-','*-']
    for i in range(len(car_routes)):
        try:
            route_i = car_routes[i]
            for route in route_i:
                x,y= [],[]
                for i in route:
                    Coordinate = CityCoordinates[i]
                    x.append(Coordinate[0])
                    y.append(Coordinate[1])
                x.append(x[0])
                y.append(y[0])
                plt.plot(x, y,shape[i], alpha=0.8, linewidth=0.8)
        except:
            print('运行失败')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plotObj(obj_list):
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('迭代次数')
    plt.ylabel('最优成本')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()

if __name__ == '__main__':
    start = tm.perf_counter()
    #农机参数
    CAPACITY = []#车辆的最大作业能力
    DISTABCE = []#车辆最大行驶距离
    C0 = []
    C1 =3
    C2 =2
    C3 =30
    V =30
    birdNum = 20
    w = random.uniform(0.4, 0.9)
    c1 = 2*random.random()
    c2 = 2*random.random()
    pBest,pLine =0,[]
    gBest,gLine = 0,[]
    iterMax = 100
    c1a =2.5
    c2a =0.5
    c1b =0.5
    c2b =2.5
    bestfit = [] 
    DC = 10 
    C = 120 
    wlist = []
    for i in range(C):
        w =random.uniform(0.6, 0.9)
        wlist.append(w)
    Customer = []
    Demand = []
    time = []
    calfit_list=[]
    dis_matrix = calDistance(Customer)
    distribution_centers = assign_distribution_center(dis_matrix,DC,C)
    bestfit_list,gLine_car_list = [],[]
    best_sumdis=[]
    end_eds=[]
    car_list=[0,0]
    len_carlist=[]
    sum_car_list=[]
    end_bestfit=[]
    for certer_number in range(len(distribution_centers)):
        distribution_center = distribution_centers[certer_number]
        birdPop = [greedy(distribution_center,dis_matrix,DC,certer_number) for i in range(birdNum)]
        birdPop_car,fits,sumdis,look_line = calFitness(birdPop,certer_number,Demand,dis_matrix,CAPACITY,DISTABCE,C0,C1,C2,C3,time,V)#分配车辆，计算种群适应度            
        gBest = pBest = min(fits)
        gLine = pLine = birdPop[fits.index(min(fits))]
        gLine_car = pLine_car = birdPop_car[fits.index(min(fits))]
        bestfit_list.append(gBest)
        calfit_list.append(gBest)
        w_list =[]
        c1_list=[]
        c2_list=[]
        end_sumdis_list=[]
        iterI = 1
        while iterI <= iterMax:
            for i in range(birdNum):
                birdPop[i],iter_w,c1,c2 = crossover(birdPop[i],pLine,gLine,w,c1,c2,wlist,c1a,c1b,c2a,c2b,fits)
                birdPop[i],r1,r2=mutation(birdPop[i])
                if min(fits)>= gBest and pBest:
                    birdPop[i][r1], birdPop[i][r2] = birdPop[i][r2], birdPop[i][r1]
            
            birdPop_car,fits,sum_distance,look_line= calFitness(birdPop,certer_number,Demand,dis_matrix,CAPACITY,DISTABCE,C0,C1,C2,C3,time,V)#分配车辆，计算种群适应度
            birdPop_car.remove(birdPop_car[fits.index(max(fits))])
            fits.remove(max(fits))
            pBest,pLine,pLine_car,end_sumdis  = min(fits),birdPop[fits.index(min(fits))],birdPop_car[fits.index(min(fits))],min(sum_distance)
            end_sumdis_list.append(end_sumdis)
            if min(fits) <= gBest:
                gBest,gLine,gLine_car,end_sumdis =  min(fits),birdPop[fits.index(min(fits))],birdPop_car[fits.index(min(fits))],min(sum_distance)
            print(iterI,gBest,min(end_sumdis_list),iter_w)
            calfit_list.append(gBest)
            iterI += 1
            end_bestfit.append(gBest)
            w_list.append(iter_w)
            c1_list.append(c1)
            c2_list.append(c2)
        best_sumdis.append(min(end_sumdis_list))
        bestfit_list.append(gBest)
        gLine_car_list.append(gLine_car)
        end_eds.append(gBest)
        plotbest =sum(end_eds)
        len_carlist.append(len(gLine_car))
        sum_car_list.append(car_list)
        
    print(gLine_car_list)
    print("最优值：",sum(end_eds))
    print("最佳行驶距离:",sum(best_sumdis))
    end = tm.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    draw_path(gLine_car_list,Customer)

  
