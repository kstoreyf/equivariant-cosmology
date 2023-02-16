import numpy as np


class FindMassRatio():
    
    def __init__(self, haloIndex, halo_mass, snap, firstprogenID, nextprogenID, index):
        self.haloIndex = haloIndex
        self.snap = snap
        self.halo_mass = halo_mass
        self.firstprogenID = firstprogenID
        self.nextprogenID = nextprogenID
        self.index = index
        self.saveFirstProgenHistory()
        
    def saveFirstProgenHistory(self):
        self.mass_history = np.array([], dtype=float)
        self.nextSnap = np.array([])
        self.snapshot_history = np.array([],dtype=int)
        self.nextID = np.array([],dtype=int)

        first_index = self.haloIndex
        while True:
            self.mass_history = np.append(self.mass_history, self.halo_mass[first_index])
            self.snapshot_history = np.append(self.snapshot_history, self.snap[first_index])
            first_ID= self.firstprogenID[first_index] 
            if first_ID == -1:
                break
            first_index = self.index[first_ID]
            if self.nextprogenID[first_index] != -1:
                self.nextID = np.append(self.nextID, self.nextprogenID[first_index])
                self.nextSnap = np.append(self.nextSnap, self.snap[first_index])
    
    def getMaxMassSnapNum(self, haloID):
        mass_history = np.array([], dtype=float)
        snapshot_history = np.array([],dtype=int)

        if haloID == self.haloIndex:
            first_index = haloID
        else:
            first_index = self.index[haloID]
        while True:
            mass_history = np.append(mass_history, self.halo_mass[first_index])
            snapshot_history = np.append(snapshot_history, self.snap[first_index])
            first_ID= self.firstprogenID[first_index]
            if first_ID == -1:
                break
            first_index = self.index[first_ID]
        mass_max = mass_history.max()
        snap_max = snapshot_history[np.where(mass_history==mass_max)[0]]
        return mass_max, snap_max[0]
    
    def getMassRatioHistory(self):
        next_history = np.array([])
        first_history = np.array([])
        mass_ratio = np.array([])
        
        ## searching for next progenitors.
        for ID in self.nextID:
            mass, snap = self.getMaxMassSnapNum(ID)
            next_history = np.append(next_history, [mass, snap]) ## next progen's max mass and snap at the time.
        next_history = next_history.reshape(-1,2)
        ## from next progen's histroy, see if there's major mergers and their mass ratio.
        for mass_next, snap_temp in next_history:
            mass_first = self.mass_history[np.where(self.snapshot_history==snap_temp)[0]]
            mass_ratio = np.append(mass_ratio, mass_next/mass_first)

        major_merger_count = len(np.where(mass_ratio > 1/4)[0])
        if major_merger_count == 0:  
            return 0, major_merger_count
        else:
            return mass_ratio[np.where(mass_ratio > 1/4)[0][0]], major_merger_count
    
    def plotMassHistory(self, ID):
        mass_history = np.array([])
        snapshot_history = np.array([])
 
        if ID == self.haloIndex:
            first_index = ID
        else :
            first_index = self.index[ID]

        while True:
            mass_history = np.append(mass_history, halo_mass[first_index])
            snapshot_history = np.append(snapshot_history, snap[first_index])

            first_ID = firstprogenID[first_index] 
            if first_ID == -1:
                break
            first_index = index[first_ID]

        plt.scatter(snapshot_history, mass_history)
        plt.show()
        plt.close()
        return 

        
def get_major_merger_count(f, index):
    nextprogenID = np.array(f['NextProgenitorID'])
    firstprogenID = np.array(f['FirstProgenitorID'])
    subhaloID = np.array(f['SubhaloID'])
    haloMass = np.array(f['SubhaloMassType'])[:,1]
    snap = np.array(f['SnapNum'])
    newID={}
    for value_t, index_t in enumerate(subhaloID):
        newID[index_t]=value_t
    total_merger_count=[]
    mass_ratio=[]
    major_merger_count=[]

    for temp in index:
        
        fmr=FindMassRatio(temp, haloMass, snap, firstprogenID, nextprogenID, newID)
        mass_ratio_t, major_merger = fmr.getMassRatioHistory()
        mass_ratio.append(mass_ratio_t)
        major_merger_count.append(major_merger)
        del fmr
        
        #############################################################
        #######################Total Merger count####################
        merge_count = 0
        ## first progenitor search
        firstprogen_index = temp

        while True:
            try:
                firstprogen_index=firstprogenID[firstprogen_index]
                if firstprogen_index == -1:
                    break
                firstprogen_index = newID[firstprogen_index]

            except IndexError:
                print(1)
                break

            nextprogen_index = firstprogen_index
            
            ## next progenitor search
            while True:
                nextprogen_index = nextprogenID[nextprogen_index]
                if (nextprogen_index == -1):
                    break
                nextprogen_index = newID[nextprogen_index]                    
                merge_count += 1

        total_merger_count.append(merge_count)
        ###########################################################
        

    return np.array(total_merger_count), np.array(mass_ratio), np.array(major_merger_count)