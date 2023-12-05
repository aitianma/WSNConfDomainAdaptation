import numpy as np
import pandas as pd
from scipy.spatial import distance

class MatSampler:

    def __init__(self,simulator="tossim"):
        #set simulator
        self.simulator=simulator

    def readData(self):
        #load simulation and physical data

        self.sim = pd.read_csv("./data/training/simulationData/%s.csv"%(self.simulator))
        self.phy = pd.read_csv("./data/training/physicalData/physicalData.csv")

    def getMahalanobisInEachNC(self):
        #get the mahalanobis distances between the centroid of  75 shots of simulation data and the 5 physical data samples

        self.readData()

        # Specify the count of shots in the simulation data
        n_sim_shots = 75
        # Specify the count of shots in the physical data
        n_phy_shots = 5
        # Specify the count of configurations
        n_configs = 88



        y_diss=[]
        res = []
        #for each network configuration
        for c in range(n_configs):
            # for each shot
            y_dis=[]
            for i in range(n_phy_shots):

                simulation_data = self.sim[self.sim["LA"]==c].loc[:, ~self.sim.columns.isin(['A', 'C','P','LA',"feature"])][0:n_sim_shots].values
                physical_sample = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][i:i+1].values

                #compute the centroid of simulation data
                centroid_sim = np.average(simulation_data,axis=0)

                #compute the max distance between centoid and the simulation data points
                cov=np.cov(np.vstack([simulation_data,physical_sample]).T)
                l = [self.Mahalanobis(xi,centroid_sim,cov=cov) for xi in simulation_data]
                max_simulation_dis = np.max(l)

                #compute the physical simulation gap
                phy_sim_dist = self.Mahalanobis(centroid_sim,physical_sample,cov=cov)
                y_dis.append(phy_sim_dist)



            y_diss.append({"%d"%(c):y_dis})
            res.append(y_dis)

        print("--------------------------------------------------------------------------")
        print("The Mahalanobis distances between the centroid of  75 shots of simulation data and the 5 physical data samples:")
        print(y_diss)


        pd.DataFrame(res).to_csv("y_diss.csv",index=False)

        return y_diss





    def judgeByMahalanobis(self):
        #load physical and simulation Data
        self.readData()

        # Specify the count of shots in the simulation data
        n_sim_shots = 75
        # Specify the count of shots in the physical data
        n_phy_shots = 5
        # Specify the count of configurations
        n_configs = 88


        sample_count = np.ones(n_configs)*n_phy_shots

        y_diss=[]
        #for each network configuration
        for c in range(n_configs):
            # for each shot
            y_dis=[]
            for i in range(n_phy_shots):

                simulation_data = self.sim[self.sim["LA"]==c].loc[:, ~self.sim.columns.isin(['A', 'C','P','LA',"feature"])][0:n_sim_shots].values
                physical_sample = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][i:i+1].values

                #compute the centroid of simulation data
                centroid_sim = np.average(simulation_data,axis=0)

                #compute the max distance between centoid and the simulation data points
                cov=np.cov(np.vstack([simulation_data,physical_sample]).T)
                l = [self.Mahalanobis(xi,centroid_sim,cov=cov) for xi in simulation_data]
                max_simulation_dis = np.max(l)

                #compute the physical simulation gap
                phy_sim_dist = self.Mahalanobis(centroid_sim,physical_sample,cov=cov)
                y_dis.append(phy_sim_dist)

                if i<=1 :
                    # Define the minimum distance between physical data samples
                    # In the case where the number of physical samples is less than two
                    # Default to the physical simulation distance as the reference distance
                     pre_i_phy = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][0:1].values
                     physical_dists=[self.Mahalanobis(centroid_sim,pre_i_phy,cov=np.cov(np.vstack([simulation_data,pre_i_phy]).T))]

                elif i>1:
                     # Retrieve the previously recorded physical data samples
                     pre_i_phy = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][0:i].values

                     #compute the physical distances between current sample and previous samples
                     physical_dists = [self.Mahalanobis(yi,physical_sample,cov=np.cov(np.vstack([pre_i_phy,physical_sample]).T))  for yi in pre_i_phy]


                if (min(physical_dists)< abs(phy_sim_dist-max_simulation_dis) or phy_sim_dist < max_simulation_dis )  and sample_count[c]==n_phy_shots :

                    sample_count[c] = i+1
                    # break

            y_diss.append(y_dis)

        pd.DataFrame(y_diss).to_csv("y_diss.csv")

        return sample_count


    def sampleByMahalanobis(self,extraSample=0,seed=1):

        #number of configurations
        n_configs = 88
        #get sample count in each network configuration
        sample_count = self.judgeByMahalanobis()

        #adding or removing extra sample
        print("Algorithm selected sample count:",sample_count)

        print(np.sum(sample_count))
        #concat all the samples in each configuration
        sample = []
        remaining = []
        for c in range(n_configs):
            item = pd.DataFrame(self.phy[self.phy["LA"]==c][0:int(sample_count[c])],columns=[ "A","C","P","B","L","R","LA"]).values
            sample.append(item)
            remaining.append(pd.DataFrame(self.phy[self.phy["LA"]==c][int(sample_count[c]):5],columns=[ "A","C","P","B","L","R","LA"]).values)

        sample = np.vstack(sample)
        remaining = np.vstack(remaining)

        #shuffle sample and remaining dataset so that we could randomly add or remove samples from physical dataset
        np.random.shuffle(sample)
        np.random.shuffle(remaining)


        if extraSample>0:
            sample = np.vstack([sample,remaining[0:extraSample]])
        else:
            sample = sample[0:len(sample)+extraSample]



        #save the sample to file
        pd.DataFrame(sample,columns=[ "A","C","P","B","L","R","LA"]).to_csv("./data/training/physicalDataWithSamplingAlgorithm/%s_selected_samples.csv"%(self.simulator),index=False)

        pd.DataFrame(sample,columns=[ "A","C","P","B","L","R","LA"]).to_csv("./data/intermediate/%s_seed_%d_selected_%d_samples.csv"%(self.simulator,seed,len(sample)),index=False)

        #return the number of samples
        return len(sample)



    def Mahalanobis(self,y=None, data=None, cov=None):

        # use Mahalanobis function to calculate
        # the Mahalanobis distance


        if cov is  None:
            cov = np.cov(np.array(data).T)

        try:
            inv_covmat = np.linalg.inv(cov)
        except:
            inv_covmat = np.linalg.pinv(cov)


        mahal = distance.mahalanobis(y,data,inv_covmat)

        return mahal

